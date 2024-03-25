import jax
import jax.numpy as jnp
import time

import torch


def test_jax_matmul():
    x = jax.random.normal(jax.random.PRNGKey(0), (10, 10))
    y = jax.random.normal(jax.random.PRNGKey(1), (10, 10))
    
    x = jax.device_put(x)
    y = jax.device_put(y)

    print('JAX dtype:',jnp.dtype(x))
    # Warm up (JIT compilation happens during the first call)
    x@y.block_until_ready()

    # Measure performance
    start_time = time.time()
    for _ in range(100):
        x@y
    jax_duration = time.time() - start_time

    print(f"JAX matmul Duration: {jax_duration} seconds")



def test_torch_matmul():
    x = torch.randn(10, 10,dtype=torch.float32).cuda()
    y = torch.randn(10, 10,dtype=torch.float32).cuda()
    print('PyTorch dtype:',x.dtype)

    start_time = time.time()
    for _ in range(100):
        x@y
    torch_duration = time.time() - start_time
    print(f"PyTorch matmul Duration: {torch_duration} seconds")

if __name__ =='__main__':
    test_jax_matmul()
    test_torch_matmul()