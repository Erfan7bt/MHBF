import numpy as np
import matplotlib.pyplot as plt

def visCov(model):
    """
    Visualize the modified triangular covariance matrix of the model. 

    Ignoring last row and first column.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to visualize.
    """
    rank = model.rank
    if rank == 1:
        wi= model.wi.detach().numpy().squeeze()
        w= model.w.detach().numpy().squeeze()
        n= model.n.detach().numpy().squeeze()
        m= model.m.detach().numpy().squeeze()

        vectors = [wi, n, m, w]
        names = ['I', 'n', 'm', 'w']

        cov=np.cov(vectors)
        cov=np.triu(cov, k=1)
        cov=cov[:-1,1:]
        bound=bound = np.max((np.abs(np.min(cov)), np.abs(np.max(cov))))

        fig, ax = plt.subplots()

        im = ax.imshow(cov, cmap='RdBu', vmin=-bound, vmax=bound)
        
        ax.xaxis.tick_top()
        ax.yaxis.tick_right()
        ax.set_aspect('equal','box')

        plt.xticks(np.arange(0, 3, step=1), names[1:], fontsize=20)
        plt.yticks(np.arange(0, 3, step=1), names[:-1], fontsize=20)
        plt.colorbar()
        plt.show()
    
    elif rank == 2:
        m=model.m.detach().numpy()
        m1=m[:,0]
        m2=m[:,1]


        n=model.n.detach().numpy()
        n1=n[:,0]
        n2=n[:,1]


        wi=model.wi.detach().numpy()

        w=model.w.detach().numpy()
        w=w.squeeze()



        vector=[wi,n1,n2,m1,m2,w]
        names=['I','n1','n2','m1','m2','w']

        cov=np.cov(vector)
        cov=np.triu(cov,k=1)
        cov=cov[:-1,1:]
        bound=np.max((np.abs(np.min(cov)),np.abs(np.max(cov))))

        fig, ax = plt.subplots()
        im = ax.imshow(cov, cmap='RdBu', vmin=-bound, vmax=bound)
        
        ax.xaxis.tick_top()
        ax.yaxis.tick_right()
        ax.set_aspect('equal','box')

        plt.xticks(np.arange(0, 5, step=1), names[1:], fontsize=20)
        plt.yticks(np.arange(0, 5, step=1), names[:-1], fontsize=20)
        plt.colorbar()
        plt.show()


