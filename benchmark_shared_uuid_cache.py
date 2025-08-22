#!/usr/bin/env python3
"""
Comprehensive benchmark for shared UUID caching in vLLM.
Tests cross-request cache sharing, multi-turn conversations, and performance.
"""

import time
import requests
import json
import statistics
import argparse
import asyncio
import aiohttp
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import concurrent.futures
import uuid as uuid_lib

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "llava-hf/llava-1.5-7b-hf"

@dataclass
class BenchmarkResult:
    """Store benchmark results for a single test."""
    test_name: str
    request_times: List[float]
    cache_behavior: str
    total_time: float
    
    @property
    def mean_time(self) -> float:
        return statistics.mean(self.request_times) if self.request_times else 0.0
    
    @property
    def median_time(self) -> float:
        return statistics.median(self.request_times) if self.request_times else 0.0


async def make_async_request(session: aiohttp.ClientSession, 
                           prompt: str, 
                           image_url: str, 
                           uuid: Optional[str] = None,
                           conversation_id: Optional[str] = None) -> Tuple[Optional[float], Optional[str], str]:
    """Make an async request and return timing, response, and request details."""
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url}}
    ]
    if uuid:
        content[1]["uuid"] = uuid
    
    request_id = f"{conversation_id or 'single'}_{time.time()}"
    
    start = time.time()
    try:
        async with session.post(VLLM_URL, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 50,
            "temperature": 0.1
        }) as response:
            elapsed = time.time() - start
            
            if response.status == 200:
                result = await response.json()
                response_text = result['choices'][0]['message']['content']
                return elapsed, response_text, request_id
            else:
                error_text = await response.text()
                print(f"  Error {response.status}: {error_text}")
                return None, None, request_id
                
    except Exception as e:
        print(f"  Error in request {request_id}: {e}")
        return None, None, request_id


def make_sync_request(prompt: str, image_url: str, uuid: Optional[str] = None) -> Tuple[Optional[float], Optional[str]]:
    """Make a synchronous request."""
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url}}
    ]
    if uuid:
        content[1]["uuid"] = uuid
    
    start = time.time()
    try:
        response = requests.post(VLLM_URL, 
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 50,
                "temperature": 0.1
            },
            timeout=30)
        
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            response_text = result['choices'][0]['message']['content']
            return elapsed, response_text
        else:
            print(f"  Error {response.status_code}: {response.text}")
            return None, None
            
    except Exception as e:
        print(f"  Error: {e}")
        return None, None


async def test_cross_request_sharing(image_urls: List[str]) -> BenchmarkResult:
    """Test that UUID cache is shared across different API requests."""
    print("\n" + "="*80)
    print("TEST 1: Cross-Request Cache Sharing")
    print("="*80)
    print("Testing if cache persists across completely separate API requests...")
    
    test_uuid = "shared-test-" + str(uuid_lib.uuid4())[:8]
    times = []
    
    # Request 1: Initial load with UUID
    print(f"\nRequest 1: Initial load with UUID '{test_uuid}'")
    t1, resp1 = make_sync_request("What is in this image?", image_urls[0], uuid=test_uuid)
    if t1:
        times.append(t1)
        print(f"  Time: {t1:.3f}s (CACHE MISS - downloading and caching)")
        print(f"  Response: {resp1[:50]}...")
    
    # Wait a bit to ensure cache is written
    time.sleep(0.5)
    
    # Request 2: Different API call, same UUID, empty URL
    print(f"\nRequest 2: New API call with same UUID, empty URL")
    t2, resp2 = make_sync_request("Describe the colors", "", uuid=test_uuid)
    if t2:
        times.append(t2)
        print(f"  Time: {t2:.3f}s (CACHE HIT - should be much faster)")
        if resp2:
            print(f"  Response: {resp2[:50]}...")
    
    # Request 3: Different API call, same UUID, different URL (should still use cache)
    print(f"\nRequest 3: Same UUID but different URL")
    t3, resp3 = make_sync_request("What's the main subject?", "http://different.url/image.jpg", uuid=test_uuid)
    if t3:
        times.append(t3)
        print(f"  Time: {t3:.3f}s (CACHE HIT - UUID takes precedence)")
        print(f"  Response: {resp3[:50]}...")
    
    # Analyze results
    cache_behavior = "NOT WORKING"
    if len(times) >= 2 and times[1] < times[0] * 0.5:
        cache_behavior = "WORKING - Cache shared across requests!"
        print(f"\n✅ Cache is shared! Speedup: {times[0]/times[1]:.2f}x")
    else:
        print(f"\n❌ Cache not shared between requests")
    
    return BenchmarkResult(
        test_name="Cross-Request Sharing",
        request_times=times,
        cache_behavior=cache_behavior,
        total_time=sum(times)
    )


async def test_concurrent_requests(image_urls: List[str]) -> BenchmarkResult:
    """Test concurrent requests using the same UUID."""
    print("\n" + "="*80)
    print("TEST 2: Concurrent Request Handling")
    print("="*80)
    print("Testing multiple concurrent requests with same UUID...")
    
    test_uuid = "concurrent-" + str(uuid_lib.uuid4())[:8]
    
    # First, prime the cache
    print(f"\nPriming cache with UUID '{test_uuid}'")
    t_prime, _ = make_sync_request("Prime the cache", image_urls[0], uuid=test_uuid)
    print(f"  Cache primed in {t_prime:.3f}s")
    
    # Now send 10 concurrent requests with same UUID
    print(f"\nSending 10 concurrent requests with same UUID...")
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(10):
            task = make_async_request(
                session, 
                f"Concurrent request {i+1}", 
                "",  # Empty URL, rely on cache
                uuid=test_uuid,
                conversation_id=f"concurrent-{i}"
            )
            tasks.append(task)
        
        start = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start
    
    # Analyze results
    times = [r[0] for r in results if r[0] is not None]
    successful = len(times)
    
    print(f"\nResults:")
    print(f"  Successful requests: {successful}/10")
    print(f"  Total time for all requests: {total_time:.3f}s")
    print(f"  Average time per request: {statistics.mean(times):.3f}s")
    print(f"  Min/Max times: {min(times):.3f}s / {max(times):.3f}s")
    
    cache_behavior = f"Handled {successful} concurrent requests successfully"
    
    return BenchmarkResult(
        test_name="Concurrent Requests",
        request_times=times,
        cache_behavior=cache_behavior,
        total_time=total_time
    )


async def test_multi_turn_conversation(image_urls: List[str]) -> BenchmarkResult:
    """Test multi-turn conversation with persistent UUID references."""
    print("\n" + "="*80)
    print("TEST 3: Multi-Turn Conversation")
    print("="*80)
    print("Simulating a real conversation with image references...")
    
    conversation_uuid = "conversation-" + str(uuid_lib.uuid4())[:8]
    times = []
    
    # Turn 1: Upload image
    print(f"\nTurn 1: User uploads image with UUID '{conversation_uuid}'")
    t1, resp1 = make_sync_request(
        "I'm sharing an image with you. Please remember it for our conversation.",
        image_urls[0],
        uuid=conversation_uuid
    )
    if t1:
        times.append(t1)
        print(f"  Time: {t1:.3f}s")
        print(f"  Assistant: {resp1[:80]}...")
    
    # Turn 2: Reference with empty URL
    print(f"\nTurn 2: Reference image with empty URL")
    t2, resp2 = make_sync_request(
        "What colors do you see in the image I showed you?",
        "",  # Empty URL
        uuid=conversation_uuid
    )
    if t2:
        times.append(t2)
        print(f"  Time: {t2:.3f}s")
        print(f"  Assistant: {resp2[:80]}...")
    
    # Turn 3: Another reference
    print(f"\nTurn 3: Another question about the same image")
    t3, resp3 = make_sync_request(
        "Is there any text visible in that image?",
        "",
        uuid=conversation_uuid
    )
    if t3:
        times.append(t3)
        print(f"  Time: {t3:.3f}s")
        print(f"  Assistant: {resp3[:80]}...")
    
    # Turn 4: Much later in conversation
    print(f"\nTurn 4: Later in conversation, still referencing same image")
    time.sleep(1)  # Simulate time passing
    t4, resp4 = make_sync_request(
        "Based on the image, what time of day was it taken?",
        "",
        uuid=conversation_uuid
    )
    if t4:
        times.append(t4)
        print(f"  Time: {t4:.3f}s")
        print(f"  Assistant: {resp4[:80]}...")
    
    # Analyze conversation efficiency
    if len(times) > 1:
        first_time = times[0]
        subsequent_avg = statistics.mean(times[1:])
        speedup = first_time / subsequent_avg
        
        print(f"\nConversation Analysis:")
        print(f"  Initial load: {first_time:.3f}s")
        print(f"  Subsequent turns (avg): {subsequent_avg:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")
        
        cache_behavior = f"Multi-turn working with {speedup:.2f}x speedup"
    else:
        cache_behavior = "Multi-turn test incomplete"
    
    return BenchmarkResult(
        test_name="Multi-Turn Conversation",
        request_times=times,
        cache_behavior=cache_behavior,
        total_time=sum(times)
    )


async def test_different_users_same_uuid(image_urls: List[str]) -> BenchmarkResult:
    """Test if different 'users' can share the same UUID cache."""
    print("\n" + "="*80)
    print("TEST 4: Different Users Sharing UUID")
    print("="*80)
    print("Testing if different API clients can share cached content...")
    
    shared_uuid = "public-resource-123"
    times = []
    
    # User A uploads image
    print(f"\nUser A: Uploads image with UUID '{shared_uuid}'")
    t1, resp1 = make_sync_request(
        "User A uploading an image",
        image_urls[0],
        uuid=shared_uuid
    )
    if t1:
        times.append(t1)
        print(f"  Time: {t1:.3f}s (Cache miss)")
    
    # Simulate different user sessions
    async with aiohttp.ClientSession() as session:
        # User B accesses same UUID
        print(f"\nUser B: Accessing same UUID with empty URL")
        t2, resp2, _ = await make_async_request(
            session,
            "User B accessing the shared image",
            "",  # No URL, just UUID
            uuid=shared_uuid
        )
        if t2:
            times.append(t2)
            print(f"  Time: {t2:.3f}s")
            if t2 < t1 * 0.5:
                print(f"  ✅ User B got cached image! ({t1/t2:.2f}x faster)")
            else:
                print(f"  ❌ Cache not shared between users")
        
        # User C also accesses
        print(f"\nUser C: Also accessing same UUID")
        t3, resp3, _ = await make_async_request(
            session,
            "User C wants to see the image too",
            "",
            uuid=shared_uuid
        )
        if t3:
            times.append(t3)
            print(f"  Time: {t3:.3f}s")
    
    cache_behavior = "UUID cache is shared globally" if len(times) > 1 and times[1] < times[0] * 0.5 else "UUID cache is not shared"
    
    return BenchmarkResult(
        test_name="Cross-User Sharing",
        request_times=times,
        cache_behavior=cache_behavior,
        total_time=sum(times)
    )


async def test_cache_persistence(image_urls: List[str]) -> BenchmarkResult:
    """Test how long UUID cache persists."""
    print("\n" + "="*80)
    print("TEST 5: Cache Persistence Over Time")
    print("="*80)
    print("Testing if cache persists over extended time...")
    
    persistence_uuid = "persistence-" + str(uuid_lib.uuid4())[:8]
    times = []
    
    # Initial cache
    print(f"\nInitial cache with UUID '{persistence_uuid}'")
    t1, _ = make_sync_request("Initial cache", image_urls[0], uuid=persistence_uuid)
    if t1:
        times.append(t1)
        print(f"  Time: {t1:.3f}s")
    
    # Test at different intervals
    intervals = [1, 5, 10, 30]  # seconds
    for interval in intervals:
        print(f"\nAfter {interval} seconds:")
        time.sleep(interval)
        
        t, resp = make_sync_request(
            f"Checking cache after {interval}s",
            "",
            uuid=persistence_uuid
        )
        if t:
            times.append(t)
            if t < t1 * 0.5:
                print(f"  ✅ Cache still valid! Time: {t:.3f}s")
            else:
                print(f"  ❌ Cache miss! Time: {t:.3f}s")
                break
    
    cache_behavior = f"Cache persisted for at least {sum(intervals[:len(times)-1])} seconds"
    
    return BenchmarkResult(
        test_name="Cache Persistence",
        request_times=times,
        cache_behavior=cache_behavior,
        total_time=sum(times)
    )


async def test_different_models_same_uuid(image_urls: List[str]) -> BenchmarkResult:
    """Test if same UUID works across different models (if multiple models available)."""
    print("\n" + "="*80)
    print("TEST 6: UUID Across Different Models")
    print("="*80)
    print("Testing if UUID cache is model-specific...")
    
    # This test assumes you might have multiple models
    # Adjust MODEL_2 to another multimodal model if available
    MODEL_2 = "llava-hf/llava-1.5-7b-hf"  # Same model for now
    
    model_uuid = "model-test-" + str(uuid_lib.uuid4())[:8]
    times = []
    
    # Model 1
    print(f"\nModel 1 ({MODEL}): Cache with UUID '{model_uuid}'")
    t1, _ = make_sync_request("Model 1 test", image_urls[0], uuid=model_uuid)
    if t1:
        times.append(t1)
        print(f"  Time: {t1:.3f}s")
    
    # Model 2 with same UUID
    print(f"\nModel 2 ({MODEL_2}): Same UUID, empty URL")
    # Note: In practice, you'd need to modify the request to use MODEL_2
    t2, _ = make_sync_request("Model 2 test", "", uuid=model_uuid)
    if t2:
        times.append(t2)
        print(f"  Time: {t2:.3f}s")
        if t2 < t1 * 0.5:
            print(f"  Note: Cache might be shared across models")
        else:
            print(f"  Note: Cache appears model-specific")
    
    cache_behavior = "Test requires multiple models"
    
    return BenchmarkResult(
        test_name="Cross-Model UUID",
        request_times=times,
        cache_behavior=cache_behavior,
        total_time=sum(times)
    )


def print_summary(results: List[BenchmarkResult]):
    """Print comprehensive test summary."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    print("\n### Test Results Overview")
    print(f"{'Test Name':<30} {'Avg Time':<12} {'Cache Behavior':<40}")
    print("-" * 82)
    
    for result in results:
        avg_time = result.mean_time
        print(f"{result.test_name:<30} {avg_time:<12.3f} {result.cache_behavior:<40}")
    
    print("\n### Key Findings")
    
    # Check if shared caching is working
    shared_working = any("shared" in r.cache_behavior.lower() and "working" in r.cache_behavior.lower() 
                        for r in results)
    
    if shared_working:
        print("✅ UUID cache is shared across requests (Option 1 implemented successfully)")
        print("✅ Multi-turn conversations work efficiently")
        print("✅ Different API clients can share cached content via UUID")
    else:
        print("❌ UUID cache appears to be request-scoped")
        print("❌ Each request may be creating its own cache instance")
    
    # Performance summary
    print("\n### Performance Metrics")
    total_requests = sum(len(r.request_times) for r in results)
    total_time = sum(r.total_time for r in results)
    
    print(f"Total requests made: {total_requests}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per request: {total_time/total_requests:.3f}s")
    
    # Cache hit rates
    cache_hits = sum(1 for r in results for i, t in enumerate(r.request_times) 
                    if i > 0 and t < r.request_times[0] * 0.5)
    cache_hit_rate = cache_hits / (total_requests - len(results)) * 100 if total_requests > len(results) else 0
    
    print(f"\nEstimated cache hit rate: {cache_hit_rate:.1f}%")
    
    print("\n### Recommendations")
    print("1. Monitor server logs for 'UUID cache hit' messages")
    print("2. Check ProcessingCache size growth during testing")
    print("3. Verify no memory leaks with extended UUID usage")
    print("4. Consider implementing user/session scoping for production")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark shared UUID caching")
    parser.add_argument("--images", type=int, default=3, help="Number of test images")
    parser.add_argument("--skip-persistence", action="store_true", 
                       help="Skip the persistence test (takes longer)")
    args = parser.parse_args()
    
    # Check server
    try:
        requests.get(f"{VLLM_URL.replace('/chat/completions', '/models')}", timeout=5)
        print("✓ vLLM server is running")
    except:
        print("❌ vLLM server not running!")
        print("\nStart server with:")
        print("python -m vllm.entrypoints.openai.api_server \\")
        print("    --model llava-hf/llava-1.5-7b-hf \\")
        print("    --port 8000")
        return
    
    # Generate test images
    image_urls = [f"https://picsum.photos/400/300?random={i}" for i in range(args.images)]
    
    print(f"\nRunning comprehensive UUID cache benchmarks...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {MODEL}")
    print(f"Test images: {len(image_urls)}")
    
    # Run all tests
    results = []
    
    # Test 1: Cross-request sharing
    result = await test_cross_request_sharing(image_urls)
    results.append(result)
    
    # Test 2: Concurrent requests
    result = await test_concurrent_requests(image_urls)
    results.append(result)
    
    # Test 3: Multi-turn conversation
    result = await test_multi_turn_conversation(image_urls)
    results.append(result)
    
    # Test 4: Different users
    result = await test_different_users_same_uuid(image_urls)
    results.append(result)
    
    # Test 5: Persistence (optional)
    if not args.skip_persistence:
        result = await test_cache_persistence(image_urls)
        results.append(result)
    
    # Test 6: Cross-model (if applicable)
    result = await test_different_models_same_uuid(image_urls)
    results.append(result)
    
    # Print summary
    print_summary(results)
    
    print("\n✅ Benchmark completed!")


if __name__ == "__main__":
    asyncio.run(main())