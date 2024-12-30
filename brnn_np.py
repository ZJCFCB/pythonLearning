import numpy as np
# 输入的y为（1，n），即行向量
def softmax(y):
    y = np.exp(y)
    m = len(y)
    y_sum = y.sum(axis=1)            
    return  y/y_sum.reshape(m,1)
 
# 前向计算，得到隐藏层的 a ，a′和输出
def birnn_forward(x,a_in, ape_in, parameters_birnn):
    a_in = a_in.copy()
    ww = parameters_birnn['ww']
    wu = parameters_birnn['wu']
    bb = parameters_birnn['bb']
    wv = parameters_birnn['wv']
    
    ape_in = ape_in.copy()
    wwpe = parameters_birnn['wwpe']
    wupe = parameters_birnn['wupe']
    bbpe = parameters_birnn['bbpe']    
    wvpe = parameters_birnn['wvpe']
    
    bc = parameters_birnn['bc']
    
    m = len(x)
    a_next = np.zeros([x.shape[0],wu.shape[1]])
    ape_next = np.zeros([x.shape[0],wu.shape[1]])
       
    for i in range(m):
        a_next_i = np.tanh(a_in@ww + x[i]@wu + bb)
        ape_next_i = np.tanh(ape_in@wwpe + x[m-i-1]@wupe + bbpe)
                   
        a_next[i] = a_next_i
        ape_next[m-i-1] = ape_next_i
        
        a_in = a_next_i
        ape_in = ape_next_i
    
    y_hat = softmax(a_next@wv + ape_next@wvpe + bc)
        
    return a_next, ape_next, y_hat


# 验证前向计算
x = np.random.randn(10,27)
 
a_in = np.random.randn(1,5)
ww = np.random.randn(5,5)
wu = np.random.randn(27,5)
bb = np.random.randn(1,5)
 
ape_in = np.random.randn(1,5)
wwpe = np.random.randn(5,5)
wupe = np.random.randn(27,5)
bbpe = np.random.randn(1,5)
 
wv = np.random.randn(5,27)
wvpe = np.random.randn(5,27)
bc = np.random.randn(1,27)
 
parameters_birnn = {"ww":ww,"wu":wu,"bb":bb, "wwpe":wwpe,"wupe":wupe,"bbpe":bbpe, "wv":wv,"wvpe":wvpe,"bc":bc}
 
a_next, ape_next, y_hat = birnn_forward(x,a_in, ape_in, parameters_birnn)


#梯度
def birnn_gradient(x,a_in,ape_in,parameters_birnn,a_next,ape_next,y_hat,y):
    
  #**** 计算正向循环的梯度
    a_in = a_in.copy()
    ww = parameters_birnn['ww']
    wu = parameters_birnn['wu']
    wv = parameters_birnn['wv']
    bb = parameters_birnn['bb']
    bc = parameters_birnn['bc']    
    
    a = np.r_[a_in,a_next]   # a_in为第0个元素，a中有11行，其最大索引值为10; y_pred中共有10行，其最大索引值为9
    m = len(x)-1              # m=9
    
    # 0-9，共10个隐藏层，先计算最后一个隐藏层,即第9层的梯度
    delt_frist = (y_hat[m]-y[m]).reshape(1,-1)   # 隐藏层的初始误差
    delt_v = (a[m+1].reshape(1,-1)).T @ delt_frist
    delt_c = delt_frist
    delt_a = delt_frist @ wv.T
        
    delt_w = (a[m].reshape(1,-1)).T @ (delt_a*(1-a[m+1]**2))
    delt_u = (x[m].reshape(1,-1)).T @ (delt_a*(1-a[m+1]**2))
    delt_b = delt_a * (1-a[m+1]**2)
    
    delt_a_back = (delt_a*(1-a[m+1]**2)) @ ww.T
    
    # 计算隐藏层0-8的梯度,m=9
    for i in range(m):
        delt_frist = (y_hat[m-1-i]-y[m-1-i]).reshape(1,-1)
        delt_v += (a[m-i].reshape(1,-1)).T @ delt_frist
        delt_c += delt_frist
        delt_a = delt_frist @ wv.T + delt_a_back
        
        delt_w += (a[m-1-i].reshape(1,-1)).T @ (delt_a * (1-a[m-i]**2))
        delt_u += (x[m-1-i].reshape(1,-1)).T @ (delt_a * (1-a[m-i]**2))
        delt_b += delt_a * (1-a[m-i]**2)
        
        delt_a_back = (delt_a*(1-a[m-i]**2)) @ ww.T
        
  #**** 计算反向循环的梯度
    ape_in = ape_in.copy()
    wwpe = parameters_birnn['wwpe']
    wupe = parameters_birnn['wupe']
    wvpe = parameters_birnn['wvpe']
    bbpe = parameters_birnn['bbpe']
    
    ape = np.r_[ape_next,ape_in]           # aape_in是最底层的元素
    
     # 0-9，共10个隐藏层，先计算第0层的梯度
    delt_frist = (y_hat[0]-y[0]).reshape(1,-1)   # 隐藏层的初始误差
    delt_vpe = (ape[0].reshape(1,-1)).T @ delt_frist
    delt_ape = delt_frist @ wvpe.T
        
    delt_wpe = (ape[1].reshape(1,-1)).T @ (delt_ape*(1-ape[0]**2))
    delt_upe = (x[0].reshape(1,-1)).T @ (delt_ape*(1-a[0]**2))
    delt_bpe = delt_ape * (1-ape[0]**2)
    
    delt_ape_back = (delt_ape*(1-ape[0]**2)) @ wwpe.T
    
     # 计算隐藏层1-9的梯度
    for j in range(1,len(x)):
        delt_frist = (y_hat[j]-y[j]).reshape(1,-1)
        delt_vpe += (ape[j].reshape(1,-1)).T @ delt_frist
        delt_ape = delt_frist @ wvpe.T + delt_ape_back
        
        delt_wpe += (ape[j+1].reshape(1,-1)).T @ (delt_ape * (1-ape[j]**2))
        delt_upe += (x[j].reshape(1,-1)).T @ (delt_ape * (1-ape[j]**2))
        delt_bpe += delt_ape * (1-ape[j]**2)
        
        delt_ape_back = (delt_ape*(1-ape[j]**2)) @ wwpe.T
        
    parameters_birnn_gradient = {'delt_w':delt_w, 'delt_u':delt_u, 'delt_b':delt_b,
                               'delt_wpe':delt_wpe, 'delt_upe':delt_upe, 'delt_bpe':delt_bpe,
                               'delt_v':delt_v, 'delt_vpe':delt_vpe, 'delt_c':delt_c }
        
    return parameters_birnn_gradient

# 验证梯度
y = np.random.randn(10,27)
 
parameters_birnn_gradient = birnn_gradient(x,a_in,ape_in,parameters_birnn,a_next,ape_next,y_hat,y)