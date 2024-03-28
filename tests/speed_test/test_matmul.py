import jax
import jax.numpy as jnp
import time

import torch


def test_jax_matmul(N):
    
    x = jax.random.normal(jax.random.PRNGKey(0), (N, N))
    y = jax.random.normal(jax.random.PRNGKey(1), (N, N))
    
    x = jax.device_put(x)
    y = jax.device_put(y)
    print('JAX dtype:',jnp.dtype(x))
    # Warm up (JIT compilation happens during the first call)
    x@y.block_until_ready()

    # Measure performance
    start_time = time.time()
    for _ in range(N):
        x = jax.random.normal(jax.random.PRNGKey(0), (N, N))
        y = jax.random.normal(jax.random.PRNGKey(1), (N, N))
        
        x = jax.device_put(x)
        y = jax.device_put(y)
        (x@y).block_until_ready()
    jax_duration = time.time() - start_time

    print(f"JAX matmul Duration: {jax_duration/N} seconds")



def test_torch_matmul(N):
    
    start_time = time.time()
    for _ in range(100):
        x = torch.randn(N, N,dtype=torch.float32, device='cuda')
        y = torch.randn(N, N,dtype=torch.float32, device='cuda')
        x@y
    torch_duration = time.time() - start_time
    print(f"PyTorch matmul Duration: {torch_duration/N} seconds")

if __name__ =='__main__':
    N = 1000
    test_jax_matmul(N)
    test_torch_matmul(N)