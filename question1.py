import numpy as np
from scipy.sparse.linalg import cg

# 定義係數矩陣 A 與常數向量 b
A = np.array([
    [4, -1, 0, -1, 0, 0],
    [-1, 4, -1, 0, -1, 0],
    [0, -1, 4, 0, 1, -1],
    [-1, 0, 0, 4, -1, -1],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, 0, -1, 4]
], dtype=float)

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)

# Jacobi 方法
def jacobi(A, b, x0=None, tol=1e-10, max_iterations=1000):
    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()
    D = np.diag(A)
    R = A - np.diagflat(D)

    for i in range(max_iterations):
        x_new = (b - np.dot(R, x)) / D
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

# Gauss-Seidel 方法
def gauss_seidel(A, b, x0=None, tol=1e-10, max_iterations=1000):
    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()

    for _ in range(max_iterations):
        x_new = x.copy()
        for i in range(n):
            sigma = sum(A[i][j] * x_new[j] if j < i else A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sigma) / A[i][i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

# SOR 方法（Successive Over-Relaxation）
def sor(A, b, omega=1.25, x0=None, tol=1e-10, max_iterations=1000):
    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()

    for _ in range(max_iterations):
        x_new = x.copy()
        for i in range(n):
            sigma = sum(A[i][j] * x_new[j] if j < i else A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (1 - omega) * x[i] + (omega * (b[i] - sigma)) / A[i][i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

# 共軛梯度法（Conjugate Gradient）
def conjugate_gradient(A, b, x0=None, tol=1e-10):
    x, info = cg(A, b, x0=x0, tol=tol)
    return x

# 執行所有方法
jacobi_result = jacobi(A, b)
gs_result = gauss_seidel(A, b)
sor_result = sor(A, b)
cg_result = conjugate_gradient(A, b)

# 顯示結果
print("(a)Jacobi Method Solution:")
print(jacobi_result)

print("\n(b)Gauss-Seidel Method Solution:")
print(gs_result)

print("\n(c)SOR Method Solution:")
print(sor_result)

print("\n(d)Conjugate Gradient Method Solution:")
print(cg_result)
