from torch.utils.data import Dataset
import oneflow as flow
import torch
import numpy as np

# a = [{"text":np.random.rand(2,4),
#       "label":0,
#       "position": np.random.rand(2,0)},
#       {"text":np.random.rand(2,4),
#       "label":0,
#       "position": np.random.rand(2,0)}]
# b = default_collate(a)
# print(b["text"].shape)
# print(b["label"].shape)
# print(b["position"].shape)
# print(b)

#或操作
def p1():
    a = flow.randn(4,4)
    print(a)

    s1 = a>0.3
    s2 = a<=-0.2
    print(s1)
    print(s2)
    print(s1 | s2)

#索引操作
def p2():
    a = np.random.rand(4,4)

    oi = flow.Tensor(a)
    ms = oi>0.5
    
    print(oi[ms].shape)

    oi[ms] = 0
    print(oi)


#torch和oneflow int32索引不一致
def p3():
    n = np.random.rand(3,4)

    ti = torch.tensor(n)
    mask = ti>0.2
    ti[mask] = 0
    print(ti[mask].shape)

    oi = flow.Tensor(n)
    mask = oi>0.2
    mask = mask.to(flow.int32)
    oi[mask] = 0
    print(oi[mask].shape)

#nn.functional.embedding函数缺失
def p4():
    from oneflow.nn.parameter import Parameter
    class a(flow.nn.Module):
        def __init__(self) -> None:
            super(a,self).__init__()
            self.weight = Parameter(flow.Tensor(1000,200))
        def forward(self,input):
            out = flow.nn.functional.embedding(
                input,
                self.weight,
                None,
                None,
                2,
                False,
                False
                )
            return out
    model = a()
    input = flow.randint(0,20,(30,50))
    out = model(input)
    print(out.shape)


#flow.numel函数缺失
def p5():
    a = np.random.rand(3,4)

    ti = torch.Tensor(a)
    print(torch.numel(ti))

    oi = flow.Tensor(a)
    print(flow.numel(oi))

#tensor.expand_as 缺失
def p6():
    print("aaa")
    m = flow.randn(8,256,256)
    ids = flow.randn(8,1,256)
    mask = ids < 0.5
    mask = mask.expand_as(m)
    

#nn.functional.linear()支持一下
def p7(o_i, o_w, o_b):
    if o_i.dim() == 2 and o_b is not None:
        res = flow.addmm(o_b,o_i,flow._C.transpose(o_w,(1,0)))
    else:
        res = flow._C.matmul(o_i, o_w, transpose_a=False, transpose_b=True)
        if o_b is not None:
            res += o_b
    return res


a = np.random.randint(-10,10,(8,256))
b = np.random.rand(8,256)
for i in range(8):
    for j in range(256):
        b[i][j]=0
print(b.shape)

ia = flow.Tensor(a).to(flow.int).cuda()
ib = flow.Tensor(b)
ib = (ib > 0.5).to(flow.int8).cuda()

s = ia[0,0]
print(s)
print(ia)
print(ib)
ia[ib] = 0
print(ia)