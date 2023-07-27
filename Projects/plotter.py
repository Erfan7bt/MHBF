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
    
    plt.xticks(np.arange(0, num_ticks, step=1), names[1:]) #, fontsize=fontsize)
    plt.yticks(np.arange(0, num_ticks, step=1), names[:-1]) # , fontsize=fontsize)

    cbar = plt.colorbar(mappable=im, pad=0.1)
    # cbar.ax.tick_params(labelsize=fontsize)
    
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


def plot_input_range(model, f_in_vec, in_params, 
                     num_repeat=10,
                     alpha=0,
                     figsize=(8,6),
                     cmap_str='turbo',
                     background='grey',
                     linewidth=2.0):
    
    if alpha==0:
        alpha = 1/num_repeat
            
    cmap = plt.cm.get_cmap(cmap_str, len(in_params))
    tmp_u, tmp_y = f_in_vec(in_params[0])
 
    plt.figure(figsize=figsize)
    plt.xlabel("Time step")
    plt.ylabel("Network output (z)")
    plt.title("Each input repeated {} times".format(num_repeat))
    
    #ax = plt.axes()
    #ax.set_facecolor("grey")

    for idx, par in enumerate(in_params):
        
        data = np.zeros((num_repeat, tmp_u.shape[0]))
        
        for i in range(num_repeat):
            data[i], y = f_in_vec(par)
            
        data = torch.Tensor(data)
            
        z = model(data).detach().numpy()
        plt.plot(z[0].T, c=cmap(idx), 
                 label=str(par), alpha=alpha,
                 linewidth=linewidth)
        if num_repeat > 1:
            plt.plot(z[1:].T, c=cmap(idx), 
                     alpha=alpha, linewidth=linewidth)
            
    ax = plt.gca()
    ax.set_facecolor(background)
    
    plt.legend(loc='right')
    plt.show()


def plot_neuron_states(model, u, y,
                       figsize=(8, 6),
                      alpha=0.5,
                      apply_activation=False,
                      plot_by='',
                      linewidth=1.5):
    """
    The weights are normalized by their max absolute value and used to control
    the alpha parameter of each line.

    Param:
    plot_by
        'weight' - Colors neurons with positive weights blue and negative red
        'adj_weight' - Each neuron is multiplied by the sign of its weight.
    """

    cmap = plt.cm.get_cmap('brg', model.network_size)

    plt.figure(figsize=figsize)
    plt.xlabel("Time step")
    plt.ylabel("Neuron value")
    if apply_activation:
        plt.ylabel("Neuron activation")
    plt.title("")

    z, activity = model(u, visible_activity=True)
    if apply_activation:
        activity = model.activation(activity)
    activity = activity.detach().numpy()
    activity = activity.squeeze()
    w = model.w.detach().numpy().squeeze()
    max_abs_w = np.abs(w).max()
    w /= max_abs_w

    print(u.shape)

    for i in range(u.size(0)):

        color = cmap(i)
        act = activity.T[i]

        if plot_by == 'weight':
            color = 'b'
            if w[i] < 0: color = 'r'

        if plot_by == 'adj_weight':
            if w[i] < 0: act *= -1
            color = 'b'
            if act[-1] < 0: color = 'r'

        plt.plot(act, alpha=abs(w[i]), c=color, linewidth=linewidth)


def plot_network_in_m_i(model, f_in_vec, in_params, 
                        num_repeat=10, 
                        figsize=(8,6),
                        alpha=0,
                        background='grey',
                        linewidth=2.5):
    
    if alpha==0:
        alpha = 1/num_repeat
    
    cmap = plt.cm.get_cmap('turbo', len(in_params))

    plt.figure(figsize=figsize)
    plt.xlabel("m component")
    plt.ylabel("I component")
    
    
    wi = model.wi.detach().numpy().squeeze()
    m = model.m.detach().numpy().squeeze()
    
    wi_dot_wi = wi.dot(wi)
    m_dot_m = m.dot(m)
    
    wi_norm = wi / wi_dot_wi**0.5
    m_norm = m / m_dot_m**0.5
    
    m_dot_wi = m_norm.dot(wi_norm)
    
    plt.title("$m \cdot I = {:1.5f}$".format(m_dot_wi))
    
    
    for idx, par in enumerate(in_params):
        for i in range(num_repeat):
            u, _ = f_in_vec(par)
            u = torch.tensor(u, dtype=torch.float32)
            z, unit_activity = model(u, visible_activity=True)
            unit_activity = unit_activity.detach().numpy().squeeze()

            wi_comp = (wi @ unit_activity.T) / wi_dot_wi
            m_comp = (m @ unit_activity.T) / m_dot_m
            
            if i == 0:
                plt.plot(m_comp, wi_comp, 
                         color=cmap(idx), alpha=alpha, 
                         label=str(par), linewidth=linewidth)
            else:
                plt.plot(m_comp, wi_comp, 
                         color=cmap(idx), alpha=alpha,
                         linewidth=linewidth)

            plt.scatter(m_comp[0], wi_comp[0], color='g')
            plt.scatter(m_comp[-1], wi_comp[-1], color='r')
            
    ax = plt.gca()
    ax.set_facecolor(background)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
