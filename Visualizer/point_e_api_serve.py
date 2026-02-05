"""
Flask API Server for Point-E to SDF Pipeline
Uses LOCAL Point-E models + Automatic Ollama management
Run with: python flask_server.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import numpy as np
import json
import os
from scipy.spatial import cKDTree
import gc
import sys
from multiprocessing import Queue
from datetime import datetime

# Point-E imports
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config


CUDA_ROOT = r"root\cuda\v13.0"
CUDA_BIN  = os.path.join(CUDA_ROOT, "bin")

os.environ["CUDA_PATH"] = CUDA_ROOT
os.environ["PATH"] = CUDA_BIN + ";" + os.environ.get("PATH", "")

from AI_Model.log import log
from AI_Model.llm_wrapper import DEFAULT_MODEL , preload_model
from AI_Model.tools.sub_llm.sub_llm_wrapper import DEFAULT_MODEL as SUB_DEFAULT_MODEL

# Ollama management
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# ============================================================================
# OLLAMA MANAGER
# ============================================================================

class OllamaManager:
    """Manages Ollama to free/reclaim VRAM"""
    
    def __init__(self):
        self.ollama_was_running = False
        self.active_models = []
    
    def check_running_models(self):
        """Check which Ollama models are currently loaded"""
        if not OLLAMA_AVAILABLE:
            return []

        try:
            result = ollama.ps()
            models = []

            for m in result.get("models", []):
                # Newer Ollama uses "model"
                if "model" in m:
                    models.append(m["model"])
                # Older / alternate formats
                elif "name" in m and "tag" in m:
                    models.append(f"{m['name']}:{m['tag']}")
                elif "name" in m:
                    models.append(m["name"])

            return models

        except Exception as e:
            log(f"Could not check Ollama models: {e}", "OLLAMA")
            return []

    
    def unload_all_models(self):
        """Unload all Ollama models to free VRAM"""
        if not OLLAMA_AVAILABLE:
            log("Ollama Python module not available", "OLLAMA")
            return False
        
        try:
            log("Freeing VRAM by unloading Ollama models...", "OLLAMA")
            
            # Get currently running models
            self.active_models = [DEFAULT_MODEL, SUB_DEFAULT_MODEL]
            
            if not self.active_models:
                log("No Ollama models currently loaded", "OLLAMA")
                return True
            
            log(f"Found models: {self.active_models}", "OLLAMA")
            
            # Unload each model
            for model in self.active_models:
                try:
                    # Send empty request to unload
                    ollama.generate(model=model, prompt="", keep_alive=0)
                    log(f"Unloaded: {model}", "OLLAMA")
                except Exception as e:
                    log(f"Could not unload {model}: {e}", "OLLAMA")
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            log("VRAM freed!", "OLLAMA")
            self.ollama_was_running = len(self.active_models) > 0
            return True
            
        except Exception as e:
            log(f"Error unloading Ollama: {e}", "OLLAMA")
            return False
    
    def reload_models(self):
        """Reload previously active Ollama models"""
        if not OLLAMA_AVAILABLE or not self.ollama_was_running:
            return
        
        try:
            log("Reloading Ollama models...", "OLLAMA")
            
            for model in self.active_models:
                try:
                    # Send a small request to load the model back
                    preload_model(model=model)
                    log(f"Reloaded: {model}", "OLLAMA")
                except Exception as e:
                    log(f"Could not reload {model}: {e}", "OLLAMA")
            
            log("Ollama models reloaded!", "OLLAMA")
            
        except Exception as e:
            log(f"Error reloading Ollama: {e}", "OLLAMA")


# ============================================================================
# POINT CLOUD TO SDF CONVERTER
# ============================================================================

class PointCloudToSDF:
    def __init__(self):
        pass
    
    def estimate_sdf(self, point_cloud, query_points):
        tree = cKDTree(point_cloud)
        distances, _ = tree.query(query_points)
        sdf_values = distances.copy()
        density_threshold = np.percentile(distances, 30)
        sdf_values[distances < density_threshold] *= -1
        return sdf_values
    
    def generate_query_points(self, point_cloud, num_points=2048):
        mins = point_cloud.min(axis=0)
        maxs = point_cloud.max(axis=0)
        scale = (maxs - mins).max()
        
        uniform_points = np.random.uniform(
            mins - scale * 0.3,
            maxs + scale * 0.3,
            (num_points // 2, 3)
        )
        
        indices = np.random.choice(len(point_cloud), num_points // 2)
        surface_points = point_cloud[indices]
        noise = np.random.randn(num_points // 2, 3) * (scale * 0.1)
        near_surface = surface_points + noise
        
        query_points = np.vstack([uniform_points, near_surface])
        return query_points.astype(np.float32)


# ============================================================================
# SDF ENCODER
# ============================================================================

import torch.nn as nn

class SDFEncoder(nn.Module):
    def __init__(self, encoding_dim=64, num_shape_types=8):
        super(SDFEncoder, self).__init__()
        
        self.point_encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )
        
        self.global_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim),
        )
        
        self.shape_classifier = nn.Linear(encoding_dim, num_shape_types)
    
    def forward(self, points, sdf_values):
        x = torch.cat([points, sdf_values.unsqueeze(-1)], dim=-1)
        point_features = self.point_encoder(x)
        global_features = torch.max(point_features, dim=1)[0]
        encoding = self.global_encoder(global_features)
        shape_logits = self.shape_classifier(encoding)
        return encoding, shape_logits


# ============================================================================
# LOCAL POINT-E GENERATOR
# ============================================================================

class LocalPointEGenerator:
    """Local Point-E with automatic VRAM management"""
    
    def __init__(self, device='cuda', ollama_manager=None):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.ollama_manager = ollama_manager
        self.models_loaded = False
        
        # Don't load models yet - load on-demand
        self.base_model = None
        self.upsampler_model = None
        self.base_sampler = None
        self.upsampler_sampler = None
    
    def load_models(self):
        """Load Point-E models (called when needed)"""
        if self.models_loaded:
            return
        
        log(f"Loading Point-E models to {self.device}...", "POINT-E")
    
        if ollama_manager:
            ollama_manager.unload_all_models()
        
        # Load base model
        log("Loading base model...", "POINT-E")
        base_name = 'base40M-textvec'
        self.base_model = model_from_config(MODEL_CONFIGS[base_name], device=self.device)
        self.base_model.eval()
        base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
        
        log("Downloading checkpoints...", "POINT-E")
        self.base_model.load_state_dict(load_checkpoint(base_name, self.device))
        
        # Load upsampler
        log("Loading upsampler...", "POINT-E")
        upsampler_name = 'upsample'
        self.upsampler_model = model_from_config(MODEL_CONFIGS[upsampler_name], device=self.device)
        self.upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[upsampler_name])
        
        self.upsampler_model.load_state_dict(load_checkpoint(upsampler_name, self.device))
        
        # Create samplers
        self.base_sampler = PointCloudSampler(
            device=self.device,
            models=[self.base_model],
            diffusions=[base_diffusion],
            num_points=[1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[3.0],
            use_karras=[True],
            karras_steps=[64],
            sigma_min=[1e-3],
            sigma_max=[120],
            s_churn=[3],
        )
        
        self.upsampler_sampler = PointCloudSampler(
            device=self.device,
            models=[self.upsampler_model],
            diffusions=[upsampler_diffusion],
            num_points=[4096 - 1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[0.0],
            use_karras=[True],
            karras_steps=[64],
            sigma_min=[1e-3],
            sigma_max=[160],
            s_churn=[0],
        )
        
        self.models_loaded = True
        log("Point-E models loaded!", "POINT-E")
    
    def generate(self, prompt):
        """Generate point cloud from text"""
        log(f"Generating: '{prompt}'", "POINT-E")
        
        try:
            # Load models if not loaded
            if not self.models_loaded:
                self.load_models()
            
            # Generate base point cloud
            log("Generating base point cloud (this takes ~30 seconds)...", "POINT-E")
            samples = None
            step_count = 0
            for x in self.base_sampler.sample_batch_progressive(
                batch_size=1,
                model_kwargs=dict(texts=[prompt]),
            ):
                samples = x
                step_count += 1
                if step_count % 8 == 0:
                    log(f"Step {step_count}/64...", "POINT-E")
            
            if samples is None:
                raise Exception("Base sampling failed - no samples generated")
            
            log(f"Base point cloud generated: {samples[0].shape}", "POINT-E")
            
            # Upsample
            log("Upsampling (this takes ~30 seconds)...", "POINT-E")
            base_pc = samples[0]
            
            step_count = 0
            for x in self.upsampler_sampler.sample_batch_progressive(
                batch_size=1,
                model_kwargs=dict(low_res=base_pc.unsqueeze(0)),
            ):
                upsampled = x
                step_count += 1
                if step_count % 8 == 0:
                    log(f"Step {step_count}/64...", "POINT-E")
            
            if upsampled is None:
                raise Exception("Upsampling failed")
            
            log(f"Upsampled shape: {upsampled.shape}", "POINT-E")
            
            # Extract coordinates - shape is [batch, channels, num_points]
            coords = upsampled[0, :3, :].T.cpu().numpy()
            
            log(f"Generated {len(coords)} points", "POINT-E")
            
            if len(coords) < 100:
                raise Exception(f"Too few points generated ({len(coords)}), something went wrong")
            
            return coords
            
        except Exception as e:
            log(f"Point-E generation failed: {e}", "POINT-E")
            log("Falling back to simple geometric shape", "POINT-E")
            
            # Fallback to simple shape
            return self._generate_fallback(prompt)
    
    def _generate_fallback(self, prompt):
        """Fallback geometric shapes when Point-E fails"""
        log("Using fallback generation", "POINT-E")
        
        if any(word in prompt.lower() for word in ['sphere', 'ball', 'round', 'globe']):
            return self._generate_sphere(num_points=4096)
        elif any(word in prompt.lower() for word in ['cube', 'box', 'square']):
            return self._generate_cube(num_points=4096)
        elif any(word in prompt.lower() for word in ['cylinder', 'tube', 'can']):
            return self._generate_cylinder(num_points=4096)
        elif any(word in prompt.lower() for word in ['banana', 'curved']):
            return self._generate_banana(num_points=4096)
        else:
            return self._generate_sphere(num_points=4096)
    
    def _generate_sphere(self, radius=0.5, num_points=4096):
        """Generate sphere point cloud"""
        phi = np.random.uniform(0, 2*np.pi, num_points)
        theta = np.random.uniform(0, np.pi, num_points)
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        return np.stack([x, y, z], axis=1).astype(np.float32)
    
    def _generate_cube(self, size=0.5, num_points=4096):
        """Generate cube point cloud"""
        points = []
        points_per_face = num_points // 6
        
        for axis in range(3):
            for sign in [-1, 1]:
                face_points = np.random.uniform(-size, size, (points_per_face, 2))
                coords = [0, 0, 0]
                other_axes = [i for i in range(3) if i != axis]
                coords[other_axes[0]] = face_points[:, 0]
                coords[other_axes[1]] = face_points[:, 1]
                coords[axis] = sign * size
                points.append(np.stack(coords, axis=1))
        
        return np.vstack(points).astype(np.float32)
    
    def _generate_cylinder(self, radius=0.3, height=0.8, num_points=4096):
        """Generate cylinder point cloud"""
        theta = np.random.uniform(0, 2*np.pi, num_points)
        y = np.random.uniform(-height/2, height/2, num_points)
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        return np.stack([x, y, z], axis=1).astype(np.float32)
    
    def _generate_banana(self, num_points=4096):
        """Generate banana-like curved shape"""
        t = np.linspace(0, np.pi, num_points)
        curve_strength = 0.3
        x = np.sin(t) * 0.5
        y = (np.cos(t) - 1) * curve_strength
        z = np.zeros_like(t)
        
        radius = 0.1 * (1 - t / np.pi)
        theta = np.random.uniform(0, 2*np.pi, num_points)
        x += radius * np.cos(theta) * 0.1
        z += radius * np.sin(theta) * 0.1
        
        return np.stack([x, y, z], axis=1).astype(np.float32)
    
    def unload_models(self):
        """Unload Point-E models to free VRAM"""
        if not self.models_loaded:
            return
        
        log("Unloading Point-E models...", "POINT-E")
        
        self.base_model = None
        self.upsampler_model = None
        self.base_sampler = None
        self.upsampler_sampler = None
        self.models_loaded = False
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        log("Point-E models unloaded!", "POINT-E")
        
        # Reload Ollama
        if self.ollama_manager:
            self.ollama_manager.reload_models()


# ============================================================================
# INITIALIZE PIPELINE
# ============================================================================

log("="*60, "INIT")
log("Initializing Flask API Server", "INIT")
log("="*60, "INIT")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f"Device: {device}", "INIT")

# Initialize managers
ollama_manager = OllamaManager()

# Check Ollama status
if OLLAMA_AVAILABLE:
    running_models = ollama_manager.check_running_models()
    if running_models:
        log(f"Ollama models currently running: {running_models}", "INIT")
        log("These will be temporarily unloaded during generation", "INIT")
    else:
        log("No Ollama models currently loaded", "INIT")
else:
    log("Ollama Python module not available", "INIT")

# Initialize Point-E (don't load models yet)
pointe = LocalPointEGenerator(device=device, ollama_manager=ollama_manager)
pc_to_sdf = PointCloudToSDF()

# Load SDF encoder (small, can stay loaded)
sdf_encoder = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sdf_model_path = os.path.join(BASE_DIR, 'sdf_model_final.pth')
if os.path.exists(sdf_model_path):
    log("Loading SDF encoder...", "INIT")
    sdf_encoder = SDFEncoder(encoding_dim=64, num_shape_types=8).to(device)
    checkpoint = torch.load(sdf_model_path, map_location=device)
    sdf_encoder.load_state_dict(checkpoint['encoder'])
    sdf_encoder.eval()
    log("SDF encoder loaded!", "INIT")
else:
    log("No SDF encoder found, using simple encoding", "INIT")

log("Server ready!", "INIT")
log("="*60, "INIT")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def index():
    """Serve the HTML frontend"""
    return send_from_directory('.', 'index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'device': device,
        'sdf_encoder_loaded': sdf_encoder is not None,
        'ollama_available': OLLAMA_AVAILABLE,
        'pointe_loaded': pointe.models_loaded
    })

@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate encoding from text prompt"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({
                'success': False,
                'error': 'No prompt provided'
            }), 400
        
        log("="*60, "API")
        log(f"Request: {prompt}", "API")
        log("="*60, "API")
        
        try:
            # Step 1: Generate point cloud
            log("[1/3] Generating point cloud...", "API")
            point_cloud = pointe.generate(prompt)
            log(f"Generated {len(point_cloud)} points", "API")
            
            # Step 2: Convert to SDF
            log("[2/3] Converting to SDF...", "API")
            query_points = pc_to_sdf.generate_query_points(point_cloud)
            sdf_values = pc_to_sdf.estimate_sdf(point_cloud, query_points)
            log("Computed SDF", "API")
            
            # Step 3: Encode
            log("[3/3] Encoding...", "API")
            if sdf_encoder:
                with torch.no_grad():
                    points_tensor = torch.FloatTensor(query_points).unsqueeze(0).to(device)
                    sdf_tensor = torch.FloatTensor(sdf_values).unsqueeze(0).to(device)
                    encoding, _ = sdf_encoder(points_tensor, sdf_tensor)
                    encoding = encoding.squeeze(0).cpu().numpy()
            else:
                encoding = np.concatenate([
                    point_cloud.mean(axis=0),
                    point_cloud.std(axis=0),
                    [len(point_cloud)]
                ])
            
            log(f"Generated encoding (dim: {len(encoding)})", "API")
            
        finally:
            # Always unload Point-E after generation
            pointe.unload_models()
        
        # Return response
        response = {
            'success': True,
            'prompt': prompt,
            'encoding': encoding.tolist(),
            'point_cloud': point_cloud.tolist(),
            'metadata': {
                'num_points': len(point_cloud),
                'encoding_dim': len(encoding),
                'device': device
            }
        }
        
        log("Request completed", "API")
        
        return jsonify(response)
        
    except Exception as e:
        log(f"Error: {e}", "API")
        
        # Try to unload Point-E even on error
        try:
            pointe.unload_models()
        except:
            pass
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# RUN SERVER
# ============================================================================

def main():
    log("="*60, "SERVER")
    log("Starting Flask server on http://localhost:5000", "SERVER")
    log("="*60, "SERVER")
    log("API Endpoints:", "SERVER")
    log("  GET  /api/health   - Health check", "SERVER")
    log("  POST /api/generate - Generate encoding from prompt", "SERVER")
    log("", "SERVER")
    log("VRAM Management:", "SERVER")
    log("  - Point-E loads only when generating", "SERVER")
    log("  - Ollama automatically unloaded during generation", "SERVER")
    log("  - Ollama automatically reloaded after generation", "SERVER")
    log("="*60, "SERVER")
    
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()