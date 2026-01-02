"""
Point-E to SDF Encoding Pipeline - READY TO TEST
Combines Point-E for text→point cloud with your SDF encoder
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os

# Check if point_e is installed
try:
    from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
    from point_e.diffusion.sampler import PointCloudSampler
    from point_e.models.download import load_checkpoint
    from point_e.models.configs import MODEL_CONFIGS, model_from_config
    POINTE_AVAILABLE = True
except ImportError:
    print("⚠️  Point-E not installed. Install with: pip install point-e")
    POINTE_AVAILABLE = False

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    print("⚠️  SciPy not installed. Install with: pip install scipy")
    SCIPY_AVAILABLE = False


# ============================================================================
# POINT CLOUD TO SDF CONVERTER
# ============================================================================

class PointCloudToSDF:
    """Converts Point-E output to SDF values for training"""
    
    def __init__(self, num_sample_points=2048):
        self.num_sample_points = num_sample_points
    
    def estimate_sdf(self, point_cloud, query_points):
        """
        Estimate SDF values from a point cloud using nearest neighbor
        
        Args:
            point_cloud: [N, 3] numpy array of surface points
            query_points: [M, 3] numpy array of query locations
        
        Returns:
            sdf_values: [M] numpy array of estimated signed distances
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required. Install with: pip install scipy")
        
        # Build KD-tree for fast nearest neighbor search
        tree = cKDTree(point_cloud)
        
        # Find distance to nearest surface point
        distances, _ = tree.query(query_points)
        
        # Simple unsigned distance (we'd need normals for signed)
        sdf_values = distances.copy()
        
        # Approximate sign using density (points in dense regions = inside)
        density_threshold = np.percentile(distances, 30)
        sdf_values[distances < density_threshold] *= -1
        
        return sdf_values
    
    def generate_query_points(self, point_cloud, num_points=2048):
        """Generate smart query points around the point cloud"""
        
        # Get bounding box
        mins = point_cloud.min(axis=0)
        maxs = point_cloud.max(axis=0)
        center = (mins + maxs) / 2
        scale = (maxs - mins).max()
        
        # Sample points in and around the object
        # 50% uniform in expanded bbox
        uniform_points = np.random.uniform(
            mins - scale * 0.3,
            maxs + scale * 0.3,
            (num_points // 2, 3)
        )
        
        # 50% near surface (gaussian around point cloud points)
        indices = np.random.choice(len(point_cloud), num_points // 2)
        surface_points = point_cloud[indices]
        noise = np.random.randn(num_points // 2, 3) * (scale * 0.1)
        near_surface = surface_points + noise
        
        query_points = np.vstack([uniform_points, near_surface])
        
        return query_points.astype(np.float32)


# ============================================================================
# SDF ENCODER (copied from training code)
# ============================================================================

class SDFEncoder(nn.Module):
    """Encodes a set of 3D points + SDF values into a latent encoding"""
    
    def __init__(self, encoding_dim=64, num_shape_types=8):
        super(SDFEncoder, self).__init__()
        
        # Process each point with its SDF value
        self.point_encoder = nn.Sequential(
            nn.Linear(4, 128),  # [x, y, z, sdf]
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )
        
        # Global pooling and final encoding
        self.global_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim),
        )
        
        # Shape classifier (auxiliary task to help learning)
        self.shape_classifier = nn.Linear(encoding_dim, num_shape_types)
    
    def forward(self, points, sdf_values):
        # points: [batch, num_points, 3]
        # sdf_values: [batch, num_points]
        
        # Concatenate points with their SDF values
        x = torch.cat([points, sdf_values.unsqueeze(-1)], dim=-1)  # [batch, num_points, 4]
        
        # Encode each point
        point_features = self.point_encoder(x)  # [batch, num_points, 512]
        
        # Max pooling across points (PointNet-style)
        global_features = torch.max(point_features, dim=1)[0]  # [batch, 512]
        
        # Final encoding
        encoding = self.global_encoder(global_features)  # [batch, encoding_dim]
        
        # Shape classification (for auxiliary loss)
        shape_logits = self.shape_classifier(encoding)
        
        return encoding, shape_logits


# ============================================================================
# POINT-E WRAPPER
# ============================================================================

class PointEGenerator:
    """Wrapper for Point-E model"""
    
    def __init__(self, device='cuda'):
        if not POINTE_AVAILABLE:
            raise ImportError("point-e not installed. Install with: pip install point-e")
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Loading Point-E on {self.device}...")
        
        # Load base model
        self.base_name = 'base40M-textvec'
        self.base_model = model_from_config(MODEL_CONFIGS[self.base_name], device=self.device)
        self.base_model.eval()
        base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[self.base_name])
        
        # Load upsample model (makes point cloud denser)
        self.upsampler_name = 'upsample'
        self.upsampler_model = model_from_config(MODEL_CONFIGS[self.upsampler_name], device=self.device)
        self.upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[self.upsampler_name])
        
        # Load checkpoints
        print("📥 Downloading Point-E checkpoints (this may take a moment)...")
        self.base_model.load_state_dict(load_checkpoint(self.base_name, self.device))
        self.upsampler_model.load_state_dict(load_checkpoint(self.upsampler_name, self.device))
        
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
        
        print("✅ Point-E loaded successfully!")
    
    def generate(self, prompt, num_samples=1):
        """
        Generate point cloud from text prompt
        
        Args:
            prompt: text description (e.g., "a red motorcycle")
            num_samples: number of samples to generate
        
        Returns:
            point_clouds: list of [N, 3] numpy arrays
        """
        print(f"🎨 Generating: '{prompt}'")
        
        # Generate base point cloud
        samples = None
        for x in self.base_sampler.sample_batch_progressive(
            batch_size=num_samples,
            model_kwargs=dict(texts=[prompt] * num_samples),
        ):
            samples = x
        
        # Upsample for more detail
        point_clouds = []
        for i in range(num_samples):
            base_pc = samples[i]
            
            for x in self.upsampler_sampler.sample_batch_progressive(
                batch_size=1,
                model_kwargs=dict(low_res=base_pc.unsqueeze(0)),
            ):
                upsampled = x
            
            # Extract coordinates (ignore colors for now)
            coords = upsampled[0, :, :3].cpu().numpy()
            point_clouds.append(coords)
        
        return point_clouds


# ============================================================================
# INTEGRATED PIPELINE
# ============================================================================

class TextToSDFPipeline:
    """Complete pipeline: Text → Point Cloud → SDF Encoding"""
    
    def __init__(self, sdf_encoder_path=None, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load Point-E
        self.pointe = PointEGenerator(device=self.device)
        
        # Load your trained SDF encoder
        self.sdf_encoder = None
        if sdf_encoder_path and os.path.exists(sdf_encoder_path):
            print(f"\n🔧 Loading SDF encoder from {sdf_encoder_path}...")
            self.sdf_encoder = SDFEncoder(encoding_dim=64, num_shape_types=8).to(self.device)
            checkpoint = torch.load(sdf_encoder_path, map_location=self.device)
            self.sdf_encoder.load_state_dict(checkpoint['encoder'])
            self.sdf_encoder.eval()
            print("✅ SDF encoder loaded!")
        else:
            if sdf_encoder_path:
                print(f"⚠️  SDF encoder not found at {sdf_encoder_path}")
            print("💡 Using simple encoding (train model for better results)")
        
        self.pc_to_sdf = PointCloudToSDF()
    
    def text_to_encoding(self, prompt, save_intermediate=True):
        """
        Complete pipeline: text → encoding
        
        Args:
            prompt: text description
            save_intermediate: save point cloud and SDF data
        
        Returns:
            encoding: 64-dim vector for Three.js
            point_cloud: raw point cloud (for visualization)
        """
        print(f"\n{'='*60}")
        print(f"🚀 Processing: '{prompt}'")
        print(f"{'='*60}")
        
        # Step 1: Generate point cloud with Point-E
        print("\n[1/3] 🎨 Generating point cloud with Point-E...")
        point_clouds = self.pointe.generate(prompt, num_samples=1)
        point_cloud = point_clouds[0]
        print(f"✅ Generated {len(point_cloud)} points")
        
        # Step 2: Convert to SDF
        print("\n[2/3] 📐 Converting to SDF...")
        query_points = self.pc_to_sdf.generate_query_points(point_cloud, num_points=2048)
        sdf_values = self.pc_to_sdf.estimate_sdf(point_cloud, query_points)
        print(f"✅ Computed SDF for {len(query_points)} query points")
        
        # Step 3: Encode with your SDF encoder
        if self.sdf_encoder:
            print("\n[3/3] 🧠 Encoding with neural network...")
            with torch.no_grad():
                points_tensor = torch.FloatTensor(query_points).unsqueeze(0).to(self.device)
                sdf_tensor = torch.FloatTensor(sdf_values).unsqueeze(0).to(self.device)
                
                encoding, _ = self.sdf_encoder(points_tensor, sdf_tensor)
                encoding = encoding.squeeze(0).cpu().numpy()
            
            print(f"✅ Generated {len(encoding)}-dimensional encoding")
        else:
            # Simple encoding without neural network
            encoding = np.concatenate([
                point_cloud.mean(axis=0),  # center
                point_cloud.std(axis=0),   # scale
                [len(point_cloud)]         # complexity
            ])
            print(f"✅ Generated simple {len(encoding)}-dimensional encoding")
        
        # Save intermediate results
        if save_intermediate:
            safe_name = "".join(c for c in prompt if c.isalnum() or c in (' ', '_')).strip()[:30]
            safe_name = safe_name.replace(' ', '_')
            filename = f"output_{safe_name}.json"
            
            data = {
                'prompt': prompt,
                'point_cloud': point_cloud.tolist(),
                'query_points': query_points.tolist(),
                'sdf_values': sdf_values.tolist(),
                'encoding': encoding.tolist()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f)
            print(f"\n💾 Saved to {filename}")
        
        print(f"\n{'='*60}")
        print("✅ Done!")
        print(f"{'='*60}\n")
        
        return encoding, point_cloud


# ============================================================================
# INTERACTIVE TESTING
# ============================================================================

def interactive_test():
    """Interactive testing mode"""
    print("\n" + "="*60)
    print("🎮 POINT-E TO SDF PIPELINE - INTERACTIVE TEST")
    print("="*60)
    
    # Check dependencies
    if not POINTE_AVAILABLE:
        print("\n❌ ERROR: point-e not installed")
        print("Install with: pip install point-e")
        return
    
    if not SCIPY_AVAILABLE:
        print("\n❌ ERROR: scipy not installed")
        print("Install with: pip install scipy")
        return
    
    # Initialize pipeline
    print("\n🔧 Initializing pipeline...")
    
    # Check if trained model exists
    sdf_model_path = 'sdf_model_final.pth'
    if not os.path.exists(sdf_model_path):
        sdf_model_path = None
        print(f"⚠️  No trained SDF model found at 'sdf_model_final.pth'")
        print("💡 Using simple encoding. Train the model first for better results!")
    
    #try:
    pipeline = TextToSDFPipeline(
        sdf_encoder_path=sdf_model_path,
        device='cuda'
    )
    #except Exception as e:
        #print(f"❌ Error initializing pipeline: {e}")
        #return
    
    print("\n✅ Pipeline ready!\n")
    print("="*60)
    print("Enter prompts to generate encodings (or 'quit' to exit)")
    print("Examples: 'a red car', 'a gaming controller', 'a coffee mug'")
    print("="*60)
    
    # Interactive loop
    while True:
        prompt = input("\n💬 Enter prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not prompt:
            continue
        
        try:
            encoding, point_cloud = pipeline.text_to_encoding(prompt, save_intermediate=True)
            
            print(f"\n📊 Results:")
            print(f"   - Encoding shape: {encoding.shape}")
            print(f"   - Encoding (first 10): {encoding[:10]}")
            print(f"   - Point cloud size: {point_cloud.shape}")
            print(f"   - Point cloud range: [{point_cloud.min():.2f}, {point_cloud.max():.2f}]")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue


# ============================================================================
# BATCH TESTING
# ============================================================================

def batch_test():
    """Test multiple prompts at once"""
    print("\n" + "="*60)
    print("🎮 POINT-E TO SDF PIPELINE - BATCH TEST")
    print("="*60)
    
    # Initialize pipeline
    sdf_model_path = 'sdf_model_final.pth' if os.path.exists('sdf_model_final.pth') else None
    pipeline = TextToSDFPipeline(sdf_encoder_path=sdf_model_path, device='cuda')
    
    # Test prompts
    prompts = [
        "a red motorcycle",
        "a gaming controller",
        "a coffee mug",
        "a simple chair",
        "a banana",
        "a robot",
    ]
    
    results = []
    
    for prompt in prompts:
        try:
            encoding, point_cloud = pipeline.text_to_encoding(prompt, save_intermediate=True)
            
            results.append({
                'prompt': prompt,
                'success': True,
                'encoding_shape': encoding.shape,
                'point_cloud_shape': point_cloud.shape
            })
            
            print(f"✅ Success: {prompt}")
            
        except Exception as e:
            results.append({
                'prompt': prompt,
                'success': False,
                'error': str(e)
            })
            print(f"❌ Failed: {prompt} - {e}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 BATCH TEST SUMMARY")
    print("="*60)
    
    success_count = sum(1 for r in results if r['success'])
    print(f"\n✅ Successful: {success_count}/{len(prompts)}")
    print(f"❌ Failed: {len(prompts) - success_count}/{len(prompts)}")
    
    print("\n📁 All outputs saved to output_*.json files")
    print("🎨 Load these in your Three.js particle system!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'batch':
            batch_test()
        elif sys.argv[1] == 'interactive':
            interactive_test()
        else:
            print("Usage:")
            print("  python pointe_pipeline.py interactive  - Interactive mode")
            print("  python pointe_pipeline.py batch        - Batch test mode")
    else:
        # Default: interactive mode
        interactive_test()