import torch

def gram_schmidth_2d(v2d, w2d):
    # Normalize the first vector
    v_orthonormal = v2d / (torch.linalg.vector_norm(v2d,dim=-1, keepdim=True) + 1e-8)

    # Project w onto v_orthonormal and subtract to make w orthogonal to v
    proj = torch.linalg.vecdot(w2d, v_orthonormal).unsqueeze(-1) * v_orthonormal
    w_orthogonal = w2d - proj
    
    # Normalize the second vector
    w_orthonormal = w_orthogonal / (torch.linalg.vector_norm(w_orthogonal, dim=-1, keepdim=True) + 1e-8)

    R2d = torch.stack((v_orthonormal, w_orthonormal), dim=-1)
    RT2d = R2d.transpose(-1, -2)

    v2d = v2d.unsqueeze(-1)
    w2d = w2d.unsqueeze(-1)

    v2d_local = torch.matmul(RT2d, v2d).squeeze(-1)
    w2d_local = torch.matmul(RT2d, w2d).squeeze(-1)

    return R2d, v2d_local, w2d_local

def test():
    batch_size = 4
    v = torch.tensor([1.0, 1.0])
    v = torch.stack([v for _ in range(batch_size)])
    w = torch.tensor([-1.0, 0.0])
    w = torch.stack([w for _ in range(batch_size)])
    R, v_local, w_local = gram_schmidth_2d(v, w)
    print(f"v_local: {v_local}")
    print(f"w_local: {w_local}")
    
    print(f"R: {R}")


if __name__ == "__main__":
    test()
