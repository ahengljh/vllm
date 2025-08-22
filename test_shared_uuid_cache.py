#!/usr/bin/env python3
"""
Simple test to verify shared UUID cache is working correctly.
This test demonstrates the key improvement: cache sharing across requests.
"""

import time
import requests
import sys

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "llava-hf/llava-1.5-7b-hf"

def test_shared_cache():
    """Test that demonstrates shared UUID caching."""
    
    print("=== Shared UUID Cache Test ===\n")
    
    # Check server
    try:
        requests.get(f"{VLLM_URL.replace('/chat/completions', '/models')}", timeout=5)
        print("✓ vLLM server is running\n")
    except:
        print("❌ vLLM server not running!")
        print("\nStart server with:")
        print("python -m vllm.entrypoints.openai.api_server --model llava-hf/llava-1.5-7b-hf")
        return False
    
    # Test image and UUID
    image_url = "https://picsum.photos/400/300"
    test_uuid = "shared-cache-test-123"
    
    print(f"Test UUID: {test_uuid}")
    print(f"Test image: {image_url}\n")
    
    # Request 1: Initial load
    print("Request 1: Initial load with full URL")
    print("-" * 40)
    
    start = time.time()
    response1 = requests.post(VLLM_URL, json={
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image?"},
                {"type": "image_url", "image_url": {"url": image_url}, "uuid": test_uuid}
            ]
        }],
        "max_tokens": 50
    })
    time1 = time.time() - start
    
    if response1.status_code == 200:
        result1 = response1.json()
        print(f"✓ Success! Time: {time1:.3f}s")
        print(f"Response: {result1['choices'][0]['message']['content'][:80]}...")
    else:
        print(f"✗ Error: {response1.status_code}")
        return False
    
    # Wait a moment
    print("\nWaiting 1 second...\n")
    time.sleep(1)
    
    # Request 2: Use cached image with empty URL
    print("Request 2: Same UUID with EMPTY URL (using cache)")
    print("-" * 40)
    
    start = time.time()
    response2 = requests.post(VLLM_URL, json={
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What colors are prominent in the image?"},
                {"type": "image_url", "image_url": {"url": ""}, "uuid": test_uuid}
            ]
        }],
        "max_tokens": 50
    })
    time2 = time.time() - start
    
    if response2.status_code == 200:
        result2 = response2.json()
        print(f"✓ Success! Time: {time2:.3f}s")
        print(f"Response: {result2['choices'][0]['message']['content'][:80]}...")
    else:
        print(f"✗ Error: {response2.status_code}")
        error_text = response2.text
        print(f"Error details: {error_text}")
        
        if "not found in cache" in error_text:
            print("\n❌ CACHE NOT SHARED: UUID not found in second request")
            print("   This suggests cache is request-scoped, not shared")
            return False
    
    # Request 3: Different conversation, same UUID
    print("\nRequest 3: Completely new conversation, same UUID")
    print("-" * 40)
    
    start = time.time()
    response3 = requests.post(VLLM_URL, json={
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Is this a photo or a drawing?"},
                {"type": "image_url", "image_url": {"url": ""}, "uuid": test_uuid}
            ]
        }],
        "max_tokens": 50
    })
    time3 = time.time() - start
    
    if response3.status_code == 200:
        result3 = response3.json()
        print(f"✓ Success! Time: {time3:.3f}s")
        print(f"Response: {result3['choices'][0]['message']['content'][:80]}...")
    else:
        print(f"✗ Error: {response3.status_code}")
    
    # Analysis
    print("\n" + "="*50)
    print("RESULTS ANALYSIS")
    print("="*50)
    
    print(f"\nTiming comparison:")
    print(f"  Request 1 (cache miss): {time1:.3f}s")
    print(f"  Request 2 (cache hit?): {time2:.3f}s")
    print(f"  Request 3 (cache hit?): {time3:.3f}s")
    
    # Calculate speedup
    if time2 < time1 * 0.5 and time3 < time1 * 0.5:
        speedup2 = time1 / time2
        speedup3 = time1 / time3
        print(f"\n✅ SHARED CACHE IS WORKING!")
        print(f"   Request 2 speedup: {speedup2:.1f}x")
        print(f"   Request 3 speedup: {speedup3:.1f}x")
        print(f"\nThis proves:")
        print("- UUID cache persists across different API requests")
        print("- Empty URLs work when UUID is in cache")
        print("- Multiple conversations can share the same cached media")
        return True
    else:
        print(f"\n❌ SHARED CACHE NOT WORKING")
        print("   Subsequent requests are not significantly faster")
        print("   This suggests cache is not being shared across requests")
        return False


def test_concurrent_access():
    """Test concurrent access to same UUID."""
    print("\n\n=== Concurrent Access Test ===\n")
    
    import concurrent.futures
    
    test_uuid = "concurrent-test-456"
    image_url = "https://picsum.photos/300/200"
    
    # Prime the cache
    print("Priming cache...")
    response = requests.post(VLLM_URL, json={
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Cache this image"},
                {"type": "image_url", "image_url": {"url": image_url}, "uuid": test_uuid}
            ]
        }],
        "max_tokens": 20
    })
    
    if response.status_code != 200:
        print("Failed to prime cache")
        return False
    
    print("✓ Cache primed\n")
    
    # Define concurrent request function
    def make_request(request_id):
        start = time.time()
        response = requests.post(VLLM_URL, json={
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Request {request_id}: Describe the image"},
                    {"type": "image_url", "image_url": {"url": ""}, "uuid": test_uuid}
                ]
            }],
            "max_tokens": 30
        })
        elapsed = time.time() - start
        
        success = response.status_code == 200
        return request_id, success, elapsed
    
    # Send 5 concurrent requests
    print("Sending 5 concurrent requests with same UUID...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request, i) for i in range(1, 6)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    # Analyze results
    print("\nResults:")
    successful = 0
    total_time = 0
    for req_id, success, elapsed in sorted(results):
        status = "✓" if success else "✗"
        print(f"  Request {req_id}: {status} Time: {elapsed:.3f}s")
        if success:
            successful += 1
            total_time += elapsed
    
    print(f"\nSuccessful: {successful}/5")
    if successful == 5:
        avg_time = total_time / successful
        print(f"Average time: {avg_time:.3f}s")
        print("\n✅ All concurrent requests succeeded!")
        print("   Shared cache handles concurrent access correctly")
        return True
    else:
        print("\n❌ Some concurrent requests failed")
        return False


def main():
    """Run all tests."""
    print("Testing vLLM Shared UUID Cache Implementation")
    print("=" * 50)
    
    # Test 1: Basic shared cache
    test1_passed = test_shared_cache()
    
    # Test 2: Concurrent access
    test2_passed = test_concurrent_access()
    
    # Summary
    print("\n\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    if test1_passed and test2_passed:
        print("\n✅ ALL TESTS PASSED!")
        print("\nThe shared UUID cache is working correctly:")
        print("- Cache is shared across different API requests")
        print("- Empty URLs work when UUID is cached")
        print("- Concurrent access is handled properly")
        print("\nThis enables efficient multi-turn conversations and")
        print("cross-request media sharing via UUIDs.")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nThe UUID cache may not be properly shared.")
        print("Check the implementation in:")
        print("- vllm/entrypoints/chat_utils.py")
        print("- vllm/multimodal/processing.py")
    
    return test1_passed and test2_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)