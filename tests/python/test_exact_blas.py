import numpy as np
import pytest
from llaccel.golden import exact_blas_matmul

@pytest.mark.parametrize("m,n,k", [(1,17,16),(4,33,4096),(16,257,512),(3,1025,32)])
def test_exact_against_integer(m,n,k):
    rng=np.random.default_rng(m+n+k)
    a=rng.integers(-32768,32768,(m,k),dtype=np.int64)
    w=rng.integers(-128,128,(n,k),dtype=np.int8)
    a[0,:]=-32768; w[0,:]=-128
    np.testing.assert_array_equal(exact_blas_matmul(a,w,17),a @ w.astype(np.int64).T)

def test_reject_unsafe_accumulation():
    with pytest.raises(ValueError,match="bound"):
        exact_blas_matmul(np.array([[2**53]],dtype=np.int64),np.array([[127]],dtype=np.int8))

def test_no_float_inputs():
    with pytest.raises(ValueError,match="integer"):
        exact_blas_matmul(np.ones((1,1)),np.ones((1,1),dtype=np.int8))
