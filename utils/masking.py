import torch


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
    
class WindowAttMask():
    def __init__(self, num_patchs, window_size, device='cpu'):
        mask = torch.ones(num_patchs, num_patchs, dtype=torch.bool, requires_grad=False).to(device)
        
        for i in range(num_patchs):
            # Allow attention to elements within the window range only in the past or current position
            start = max(0, i - window_size)
            end = i + 1  # No future attention
            mask[i, start:end] = False
        self._mask = mask
    @property
    def mask(self):
        return self._mask
    
class WindowAttMaskWithRegister():
    def __init__(self, num_patchs, window_size, device, n_registers: int):
        total_size = num_patchs + n_registers
        total_mask = torch.ones(total_size, total_size, dtype=torch.bool, requires_grad=False).to(device)
        
        for i in range(num_patchs):
            # Allow attention to elements within the window range only in the past or current position
            start = n_registers + max(0, i - window_size)
            end = n_registers + i + 1  # No future attention
            total_mask[n_registers + i, start:end] = False
            
        total_mask[:, :n_registers] = False

        self._mask = total_mask
    @property
    def mask(self):
        return self._mask

class TriangularCausalSubMask():
    def __init__(self, N, M, device="cpu"):
        with torch.no_grad():
            mask = torch.ones(N, N, dtype=bool)
            
            j=M
            for i in range(N):
                mask[i:i+j, i] = False
                j -= 1
                if j == 0:
                    j = M
            self._mask = mask.to(device)

    @property
    def mask(self):
        return self._mask
    

class TriangularCausalSubLastMask():
    def __init__(self, N, M, device="cpu"):
        with torch.no_grad():
            mask = torch.ones(N, N, dtype=bool)
            
            j=M
            for i in range(N):
                mask[i:i+j, i] = False
                j -= 1
                if j == 0:
                    j = M

            for i in range(M, N + 1, M):
                for j in range(i + M, N + 1, M):
                    mask[j-1, i-1] = False
            self._mask = mask.to(device)

    @property
    def mask(self):
        return self._mask
    

class TriangularCausalSubLastBothWaysMask():
    def __init__(self, N, M, device="cpu"):
        with torch.no_grad():
            mask = torch.ones(N, N, dtype=bool)
            
            j=M
            for i in range(N):
                mask[i:i+j, i] = False
                j -= 1
                if j == 0:
                    j = M

            for i in range(M, N + 1, M):
                for j in range(M, N + 1, M):
                    mask[j-1, i-1] = False
            self._mask = mask.to(device)

    @property
    def mask(self):
        return self._mask
    

class TriangularCausalSubLastBothWaysMask():
    def __init__(self, N, M, device="cpu"):
        with torch.no_grad():
            mask = torch.ones(N, N, dtype=bool)
            
            j=M
            for i in range(N):
                mask[i:i+j, i] = False
                j -= 1
                if j == 0:
                    j = M

            for i in range(M, N + 1, M):
                for j in range(M, N + 1, M):
                    mask[j-1, i-1] = False
            self._mask = mask.to(device)

    @property
    def mask(self):
        return self._mask
    

class PatchPlusIMask():
    def __init__(self, N, M, device="cpu"):
        with torch.no_grad():
            mask = torch.ones(N, N, dtype=bool)
            
            j=M
            for i in range(N):
                mask[i:i+j, i] = False
                j -= 1
                if j == 0:
                    j = M
            for w in range(0, M):
                for i in range(w, N + 1, M):
                    for j in range(w, N + 1, M):
                        mask[j-1, i-1] = False
            self._mask = mask.to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
