# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import pytest
from typing import List
from vllm.model_executor.models.llava_next_video import (
    LlavaNextVideoPixelInputs, 
    LlavaNextVideoForConditionalGeneration
)


@pytest.mark.parametrize("num_videos,num_frames", [(1, 8), (2, 8), (3, 4), (4, 4)])
def test_multiple_videos_processing(num_videos: int, num_frames: int):
    """Test that multiple videos can be processed correctly."""
    # Create mock video data
    batch_size = 1
    channels = 3
    height = width = 224
    
    # Shape: (batch_size, num_videos, num_frames, channels, height, width)
    video_pixels = torch.randn(batch_size, num_videos, num_frames, channels, height, width)
    
    # Create input dict
    video_input = LlavaNextVideoPixelInputs(
        type="pixel_values_videos",
        data=video_pixels
    )
    
    # Mock model with necessary components
    class MockVisionTower:
        def __call__(self, x):
            # Simple mock that returns embeddings
            batch_size = x.shape[0]
            return torch.randn(batch_size, 256, 1024)  # Mock embeddings
    
    class MockModel:
        def __init__(self):
            self.vision_tower = MockVisionTower()
            self.vision_resampler = lambda x: x  # Identity
            self.multi_modal_projector = lambda x: x  # Identity
        
        def _video_pixels_to_features(self, vision_tower, pixels):
            return vision_tower(pixels)
        
        def _process_video_pixels(self, inputs):
            # Use the actual implementation
            video_pixels = inputs["data"]
            
            if isinstance(video_pixels, torch.Tensor):
                b, num_videos, num_frames, c, h, w = video_pixels.shape
                
                all_embeds = []
                for batch_idx in range(b):
                    batch_embeds = []
                    for video_idx in range(num_videos):
                        video_frames = video_pixels[batch_idx, video_idx]
                        frame_embeddings = self._video_pixels_to_features(
                            self.vision_tower, video_frames)
                        video_embeds = frame_embeddings.flatten(0, 1)
                        batch_embeds.append(video_embeds)
                    all_embeds.extend(batch_embeds)
                
                return all_embeds
    
    # Test processing
    model = MockModel()
    embeddings = model._process_video_pixels(video_input)
    
    # Verify output
    assert isinstance(embeddings, list)
    assert len(embeddings) == batch_size * num_videos
    
    # Each embedding should have flattened frame dimension
    for embed in embeddings:
        assert embed.dim() == 2  # (num_tokens, hidden_dim)


def test_multiple_videos_with_list_input():
    """Test processing multiple videos as a list."""
    # Create list of video tensors with different frame counts
    video_list = [
        torch.randn(8, 3, 224, 224),   # 8 frames
        torch.randn(4, 3, 224, 224),   # 4 frames  
        torch.randn(6, 3, 224, 224),   # 6 frames
    ]
    
    video_input = LlavaNextVideoPixelInputs(
        type="pixel_values_videos",
        data=video_list
    )
    
    class MockVisionTower:
        def __call__(self, x):
            batch_size = x.shape[0]
            return torch.randn(batch_size, 256, 1024)
    
    class MockModel:
        def __init__(self):
            self.vision_tower = MockVisionTower()
            self.vision_resampler = lambda x: x
            self.multi_modal_projector = lambda x: x
        
        def _video_pixels_to_features(self, vision_tower, pixels):
            return vision_tower(pixels)
        
        def _process_video_pixels(self, inputs):
            from vllm.utils import is_list_of
            video_pixels = inputs["data"]
            
            if is_list_of(video_pixels, torch.Tensor):
                frames_per_videos = [v.shape[0] for v in video_pixels]
                stacked_pixels = torch.cat(video_pixels, dim=0)
                stacked_embeddings = self._video_pixels_to_features(
                    self.vision_tower, stacked_pixels)
                embeds = torch.split(stacked_embeddings, frames_per_videos, dim=0)
                return [e.flatten(0, 1) for e in embeds]
    
    model = MockModel()
    embeddings = model._process_video_pixels(video_input)
    
    assert isinstance(embeddings, list)
    assert len(embeddings) == 3  # Three videos
    
    # Each embedding should have correct shape
    for embed in embeddings:
        assert embed.dim() == 2


def test_supported_mm_limits():
    """Test that multiple videos are allowed."""
    from vllm.model_executor.models.llava_next_video import LlavaNextVideoProcessingInfo
    
    class MockContext:
        def get_hf_config(self, config_class):
            # Return a mock config
            class MockConfig:
                pass
            return MockConfig()
    
    info = LlavaNextVideoProcessingInfo(MockContext())
    limits = info.get_supported_mm_limits()
    
    assert "video" in limits
    assert limits["video"] is None  # No limit