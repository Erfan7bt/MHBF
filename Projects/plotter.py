import numpy as np
import matplotlib.pyplot as plt

def visCov(model, figsize=(8,8), fontsize=20, cmap='bwr'):
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
        wi = model.wi.detach().numpy().squeeze()
        w = model.w.detach().numpy().squeeze()
        n = model.n.detach().numpy().squeeze()
        m = model.m.detach().numpy().squeeze()

        vectors = [wi, n, m, w]
        names = ['I', 'n', 'm', 'w']

        cov = np.cov(vectors)
        cov = np.triu(cov, k=1)
        cov = cov[:-1, 1:]
        bound = bound = np.max((np.abs(np.min(cov)), np.abs(np.max(cov))))
        
        num_ticks = 3        

    elif rank == 2:
        m = model.m.detach().numpy()
        m1 = m[:, 0]
        m2 = m[:, 1]

        n = model.n.detach().numpy()
        n1 = n[:, 0]
        n2 = n[:, 1]

        wi = model.wi.detach().numpy()

        w = model.w.detach().numpy()
        w = w.squeeze()

        vectors = [wi, n1, n2, m1, m2, w]
        names = ['I', 'n1', 'n2', 'm1', 'm2', 'w']
        
        num_ticks = 5
        
    cov = np.cov(vectors)
    cov = np.triu(cov, k=1)
    cov = cov[:-1, 1:]
    bound = np.max((np.abs(np.min(cov)), np.abs(np.max(cov))))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cov, cmap=cmap, vmin=-bound, vmax=bound)

    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    ax.set_aspect('equal', 'box')
    
    plt.xticks(np.arange(0, num_ticks, step=1), names[1:], fontsize=fontsize)
    plt.yticks(np.arange(0, num_ticks, step=1), names[:-1], fontsize=fontsize)

    cbar = plt.colorbar(mappable=im, pad=0.1)
    cbar.ax.tick_params(labelsize=fontsize)
    
    plt.show()

def visITO(data, model):
    """
    Visualize two instances of the Model Input-Target-Output.

    Parameters
    ----------
    data : torch.utils.data.Dataset
        Dataset to visualize.

    model : torch.nn.Module
        RNN Model to visualize.

    """

    # two random trials generated from the dataset
    d1 = data(1)
    d2 = data(1)

    u1 = d1.u.squeeze().detach().numpy()
    y1 = d1.y.squeeze().detach().numpy()

    u2 = d2.u.squeeze().detach().numpy()
    y2 = d2.y.squeeze().detach().numpy()

    z1 = model(d1.u).squeeze().detach().numpy()

    z2 = model(d2.u).squeeze().detach().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(18, 12))

    axs[0].plot(u1, label='u - input')
    axs[0].plot(y1, label='y - target')
    axs[0].set_xticks(np.arange(0, 76, step=5), np.arange(0, 1501, step=100))
    axs[0].set_xlabel('time [ms]')
    axs[0].plot(z1, label='z - model output')
    axs[0].set_title('random trial 1')
    axs[0].legend()

    axs[1].plot(u2, label='u - input')
    axs[1].plot(y2, label='y - target')
    axs[1].set_xticks(np.arange(0, 76, step=5), np.arange(0, 1501, step=100))
    axs[0].set_xlabel('time [ms]')
    axs[1].plot(z2, label='z - model output')
    axs[1].set_title('random trial 2')
    axs[1].legend()

    plt.show()


def visUA(data, model):
    """
    Visualize 3 or 5 random hidden recurrent units activity in response to the input.

    Parameters
    ----------
    data : torch.utils.data.Dataset
        Dataset to visualize.

    model : torch.nn.Modul
        RNN Model to visualize.

    """
    data = data(1)
    u = data.u
    activity = model.forward(u, visible_activity=True)[1].detach().numpy()

    rank = model.rank
    if rank == 1:
        nuits = 3
    elif rank == 2:
        nuits = 5

    plt.figure(figsize=(15, 5))
    plt.title(
        "random hidden recurrent units activity in response to the input.", fontsize=20)

    for i in range(nuits):
        unit = np.random.randint(0, model.network_size)
        unit_activity = activity[:, :, unit]
        plt.plot(unit_activity.T, label='unit {}'.format(unit))
    plt.xlabel('time [ms]')
    plt.xticks(np.arange(0, 76, step=5), np.arange(0, 1501, step=100))
    plt.ylabel('activity')
    plt.legend()


def visWP(model):
    """
    Visualize the weight vector of the model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to visualize. 

    """
    rank = model.rank
    if rank == 1:
        wi = model.wi.detach().numpy().squeeze()
        n = model.n.detach().numpy().squeeze()
        m = model.m.detach().numpy().squeeze()

        fig, ax = plt.subplots(2, 1, figsize=(5, 10))

        fig.suptitle('Connectivity Vectors projection', fontsize=20)
        # plt.suptitle('Connectivity Vectors projection', fontsize=20)
        ax[0].scatter(m, n, s=15, c='lightslategray')
        ax[0].set_xlabel('m', fontsize=14)
        ax[0].set_ylabel('n', fontsize=14)
        ax[0].xaxis.set_label_position('top')
        ax[0].yaxis.set_label_position('right')
        ax[0].xaxis.set_ticks_position('top')
        ax[0].yaxis.set_ticks_position('right')

        ax[0].spines['left'].set_position('center')
        ax[0].spines['bottom'].set_position('center')
        # Eliminate upper and right axes
        ax[0].spines['right'].set_color('none')
        ax[0].spines['top'].set_color('none')
        # Show ticks in the left and lower axes only

        ax[1].scatter(n, wi, s=15, c='lightslategray')
        ax[1].set_xlabel('n', fontsize=14)
        ax[1].set_ylabel('I', fontsize=14)
        ax[1].xaxis.set_label_position('top')
        ax[1].yaxis.set_label_position('right')
        ax[1].xaxis.set_ticks_position('top')
        ax[1].yaxis.set_ticks_position('right')

        ax[1].spines['left'].set_position('center')
        ax[1].spines['bottom'].set_position('center')
        # Eliminate upper and right axes
        ax[1].spines['right'].set_color('none')
        ax[1].spines['top'].set_color('none')
        # Show ticks in the left and lower axes only

    elif rank == 2:
        m = model.m.detach().numpy()
        m1 = m[:, 0]
        m2 = m[:, 1]
        n = model.n.detach().numpy()
        n1 = n[:, 0]
        n2 = n[:, 1]

        fig, ax = plt.subplots(2, 1, figsize=(5, 10))

        fig.suptitle('Connectivity Vectors projection', fontsize=20)
        # plt.suptitle('Connectivity Vectors projection', fontsize=20)

        ax[0].scatter(m1, n1, s=15, c='lightslategray')
        ax[0].set_xlabel('m1', fontsize=14)
        ax[0].set_ylabel('n1', fontsize=14)
        ax[0].xaxis.set_label_position('top')
        ax[0].yaxis.set_label_position('right')

        ax[0].xaxis.set_ticks_position('top')
        ax[0].yaxis.set_ticks_position('right')
        ax[0].spines['left'].set_position('center')
        ax[0].spines['bottom'].set_position('center')
        # Eliminate upper and right axes
        ax[0].spines['right'].set_color('none')
        ax[0].spines['top'].set_color('none')
        # Show ticks in the left and lower axes only

        ax[1].scatter(m2, n2, s=15, c='lightslategray')
        ax[1].set_xlabel('m2', fontsize=14)
        ax[1].set_ylabel('n2', fontsize=14)
        ax[1].xaxis.set_label_position('top')
        ax[1].yaxis.set_label_position('right')
        ax[1].xaxis.set_ticks_position('top')
        ax[1].yaxis.set_ticks_position('right')

        ax[1].spines['left'].set_position('center')
        ax[1].spines['bottom'].set_position('center')
        # Eliminate upper and right axes
        ax[1].spines['right'].set_color('none')
        ax[1].spines['top'].set_color('none')
        # Show ticks in the left and lower axes only
