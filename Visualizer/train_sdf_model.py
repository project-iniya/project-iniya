import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path

# ============================================================================
# DATASET GENERATOR - Creates synthetic SDF training data
# ============================================================================

class SDFDatasetGenerator:
    """Generates training data for various 3D shapes with SDF values"""
    
    def __init__(self, num_samples_per_shape=1000):
        self.num_samples_per_shape = num_samples_per_shape
        self.shapes = {
            'sphere': self.sphere_sdf,
            'cube': self.cube_sdf,
            'torus': self.torus_sdf,
            'cylinder': self.cylinder_sdf,
            'cone': self.cone_sdf,
            'capsule': self.capsule_sdf,
            'octahedron': self.octahedron_sdf,
            'pyramid': self.pyramid_sdf,
        }
    
    def sphere_sdf(self, points, params):
        """SDF for sphere. params: [radius]"""
        radius = params[0]
        return np.linalg.norm(points, axis=1) - radius
    
    def cube_sdf(self, points, params):
        """SDF for cube. params: [size_x, size_y, size_z]"""
        q = np.abs(points) - params
        q = np.maximum(q, 0)
        return np.linalg.norm(q, axis=1) + np.minimum(np.max(q, axis=1), 0)
    
    def torus_sdf(self, points, params):
        """SDF for torus. params: [major_radius, minor_radius]"""
        major_r, minor_r = params[0], params[1]
        q_x = np.sqrt(points[:, 0]**2 + points[:, 2]**2) - major_r
        q = np.stack([q_x, points[:, 1]], axis=1)
        return np.linalg.norm(q, axis=1) - minor_r
    
    def cylinder_sdf(self, points, params):
        """SDF for cylinder. params: [radius, height]"""
        radius, height = params[0], params[1]
        d_xz = np.sqrt(points[:, 0]**2 + points[:, 2]**2) - radius
        d_y = np.abs(points[:, 1]) - height
        return np.sqrt(np.maximum(d_xz, 0)**2 + np.maximum(d_y, 0)**2) + \
               np.minimum(np.maximum(d_xz, d_y), 0)
    
    def cone_sdf(self, points, params):
        """SDF for cone. params: [angle, height]"""
        angle, height = params[0], params[1]
        c = np.array([np.sin(angle), np.cos(angle)])
        q = np.stack([np.sqrt(points[:, 0]**2 + points[:, 2]**2), points[:, 1]], axis=1)
        q[:, 1] = -q[:, 1]
        
        # Simplified cone SDF
        w = q - c * np.maximum(np.dot(q, c), 0).reshape(-1, 1)
        return np.linalg.norm(w, axis=1)
    
    def capsule_sdf(self, points, params):
        """SDF for capsule. params: [radius, height]"""
        radius, height = params[0], params[1]
        points[:, 1] = np.abs(points[:, 1]) - height
        points[:, 1] = np.maximum(points[:, 1], 0)
        return np.linalg.norm(points, axis=1) - radius
    
    def octahedron_sdf(self, points, params):
        """SDF for octahedron. params: [size]"""
        size = params[0]
        points = np.abs(points)
        m = points[:, 0] + points[:, 1] + points[:, 2] - size
        return m * 0.57735027
    
    def pyramid_sdf(self, points, params):
        """SDF for pyramid. params: [base_size, height]"""
        base, height = params[0], params[1]
        m2 = height * height + 0.25
        
        points[:, 0] = np.abs(points[:, 0])
        points[:, 2] = np.abs(points[:, 2])
        
        # Simplified pyramid SDF
        q = np.where((points[:, 2] > points[:, 0]).reshape(-1, 1), 
                     points[:, [2, 1, 0]], points)
        
        q[:, 0] -= base * 0.5
        q[:, 2] -= base * 0.5
        
        return np.linalg.norm(q, axis=1) * 0.5
    
    def generate_random_params(self, shape_name):
        """Generate random but reasonable parameters for each shape"""
        if shape_name == 'sphere':
            return [np.random.uniform(0.3, 0.8)]
        elif shape_name == 'cube':
            return [np.random.uniform(0.2, 0.5) for _ in range(3)]
        elif shape_name == 'torus':
            major = np.random.uniform(0.4, 0.7)
            minor = np.random.uniform(0.1, 0.3)
            return [major, minor]
        elif shape_name == 'cylinder':
            return [np.random.uniform(0.2, 0.5), np.random.uniform(0.3, 0.7)]
        elif shape_name == 'cone':
            return [np.random.uniform(0.3, 0.8), np.random.uniform(0.5, 1.0)]
        elif shape_name == 'capsule':
            return [np.random.uniform(0.15, 0.3), np.random.uniform(0.3, 0.6)]
        elif shape_name == 'octahedron':
            return [np.random.uniform(0.4, 0.8)]
        elif shape_name == 'pyramid':
            return [np.random.uniform(0.4, 0.8), np.random.uniform(0.5, 1.0)]
    
    def generate_sample_points(self, num_points=2048):
        """Generate random points in 3D space for SDF evaluation"""
        # Mix of uniform sampling and surface-biased sampling
        uniform_points = np.random.uniform(-1.5, 1.5, (num_points // 2, 3))
        
        # Surface-biased: sample closer to origin
        surface_points = np.random.randn(num_points // 2, 3) * 0.5
        
        return np.vstack([uniform_points, surface_points]).astype(np.float32)
    
    def generate_dataset(self, num_shapes=5000, save_path='sdf_dataset.json'):
        """Generate complete dataset with thousands of shape variations"""
        dataset = []
        
        print(f"Generating {num_shapes} shape samples...")
        
        for i in range(num_shapes):
            # Randomly select a shape type
            shape_name = np.random.choice(list(self.shapes.keys()))
            shape_func = self.shapes[shape_name]
            
            # Generate random parameters for this shape
            params = self.generate_random_params(shape_name)
            
            # Generate sample points
            points = self.generate_sample_points(2048)
            
            # Compute SDF values at those points
            sdf_values = shape_func(points.copy(), params)
            
            # Create sample
            sample = {
                'shape_type': shape_name,
                'params': params,
                'points': points.tolist(),
                'sdf_values': sdf_values.tolist()
            }
            
            dataset.append(sample)
            
            if (i + 1) % 500 == 0:
                print(f"Generated {i + 1}/{num_shapes} samples...")
        
        # Save dataset
        print(f"Saving dataset to {save_path}...")
        with open(save_path, 'w') as f:
            json.dump(dataset, f)
        
        print(f"Dataset generated successfully! Total samples: {len(dataset)}")
        return dataset


# ============================================================================
# PYTORCH DATASET CLASS
# ============================================================================

class SDFDataset(Dataset):
    """PyTorch Dataset for loading SDF data"""
    
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Create shape type to index mapping
        shape_types = set(sample['shape_type'] for sample in self.data)
        self.shape_to_idx = {shape: idx for idx, shape in enumerate(sorted(shape_types))}
        self.idx_to_shape = {idx: shape for shape, idx in self.shape_to_idx.items()}
        
        print(f"Loaded {len(self.data)} samples")
        print(f"Shape types: {list(self.shape_to_idx.keys())}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        points = torch.FloatTensor(sample['points'])
        sdf_values = torch.FloatTensor(sample['sdf_values'])
        shape_idx = self.shape_to_idx[sample['shape_type']]
        
        # Pad params to fixed size (max 3 params for any shape)
        params = sample['params']
        params_padded = params + [0.0] * (3 - len(params))  # Pad to length 3
        params_tensor = torch.FloatTensor(params_padded)
        
        return points, sdf_values, shape_idx, params_tensor


# ============================================================================
# NEURAL NETWORK MODELS
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


class SDFDecoder(nn.Module):
    """Decodes a latent encoding + 3D coordinates into SDF value"""
    
    def __init__(self, encoding_dim=64):
        super(SDFDecoder, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(encoding_dim + 3, 256),  # [encoding, x, y, z]
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Output: SDF value
        )
    
    def forward(self, encoding, query_points):
        # encoding: [batch, encoding_dim]
        # query_points: [batch, num_query_points, 3]
        
        batch_size = encoding.shape[0]
        num_points = query_points.shape[1]
        
        # Expand encoding to match query points
        encoding_expanded = encoding.unsqueeze(1).expand(-1, num_points, -1)
        
        # Concatenate encoding with query points
        x = torch.cat([encoding_expanded, query_points], dim=-1)
        
        # Predict SDF
        sdf_pred = self.network(x).squeeze(-1)  # [batch, num_query_points]
        
        return sdf_pred


# ============================================================================
# TRAINER
# ============================================================================

class SDFTrainer:
    """Trainer for the SDF autoencoder"""
    
    def __init__(self, encoding_dim=64, num_shape_types=8, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.encoder = SDFEncoder(encoding_dim, num_shape_types).to(self.device)
        self.decoder = SDFDecoder(encoding_dim).to(self.device)
        
        # Optimizers
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=0.001
        )
        
        # Loss functions
        self.sdf_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()
        
        self.history = {'train_loss': [], 'sdf_loss': [], 'class_loss': []}
    
    def train_epoch(self, dataloader):
        self.encoder.train()
        self.decoder.train()
        
        epoch_loss = 0
        epoch_sdf_loss = 0
        epoch_class_loss = 0
        
        for batch_idx, (points, sdf_values, shape_idx, _) in enumerate(dataloader):
            points = points.to(self.device)
            sdf_values = sdf_values.to(self.device)
            shape_idx = shape_idx.to(self.device)
            
            # Forward pass - Encode
            encoding, shape_logits = self.encoder(points, sdf_values)
            
            # Forward pass - Decode
            sdf_pred = self.decoder(encoding, points)
            
            # Compute losses
            sdf_loss = self.sdf_loss(sdf_pred, sdf_values)
            class_loss = self.classification_loss(shape_logits, shape_idx)
            
            # Combined loss
            loss = sdf_loss + 0.1 * class_loss  # Weight classification loss lower
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_sdf_loss += sdf_loss.item()
            epoch_class_loss += class_loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.8f}")
        
        avg_loss = epoch_loss / len(dataloader)
        avg_sdf = epoch_sdf_loss / len(dataloader)
        avg_class = epoch_class_loss / len(dataloader)
        
        return avg_loss, avg_sdf, avg_class
    
    def train(self, dataset_path, num_epochs=50, batch_size=32):
        # Load dataset
        dataset = SDFDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Dataset size: {len(dataset)}")
        print(f"Batch size: {batch_size}")
        print("-" * 50)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            loss, sdf_loss, class_loss = self.train_epoch(dataloader)
            
            self.history['train_loss'].append(loss)
            self.history['sdf_loss'].append(sdf_loss)
            self.history['class_loss'].append(class_loss)
            
            print(f"Epoch {epoch + 1} - Total Loss: {loss:.8f}, "
                  f"SDF Loss: {sdf_loss:.8f}, Class Loss: {class_loss:.8f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'sdf_encoding_checkpoints/checkpoint_epoch_{epoch + 1}.pth')
        
        print("\nTraining completed!")
    
    def save_checkpoint(self, path):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {path}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SDF Neural Network Training System")
    print("=" * 60)
    
    # Step 1: Generate dataset
    print("\n[STEP 1] Generating dataset...")
    generator = SDFDatasetGenerator()
    generator.generate_dataset(num_shapes=5000, save_path='sdf_dataset.json')
    
    # Step 2: Train model
    print("\n[STEP 2] Training model...")
    trainer = SDFTrainer(encoding_dim=64, num_shape_types=8)
    trainer.train(dataset_path='sdf_dataset.json', num_epochs=100, batch_size=32)
    
    # Step 3: Save final model
    print("\n[STEP 3] Saving final model...")
    trainer.save_checkpoint('sdf_model_final.pth')
    
    print("\n" + "=" * 60)
    print("Training complete! Model saved as 'sdf_model_final.pth'")
    print("Dataset saved as 'sdf_dataset.json'")
    print("=" * 60)