import pprint
import matplotlib.pyplot as plt
import numpy as np

'''============================================================================
Print
============================================================================'''
def pretty_print(var):
  pp = pprint.PrettyPrinter(indent=4)
  pp.pprint(var)

'''============================================================================
Conversion
============================================================================'''
def str_to_bool(s):
    return s.strip().lower() in ['true', '1', 'yes', 'y']


'''============================================================================
Image Visualization
============================================================================'''
def multi_slice_viewer(volume:np.array,ivmin:np.float64 = None,ivmax:np.float64 = None,icmap='gray',iaxis='off'):
    if ivmin is None:
        ivmin = np.quantile(volume.flatten(),0.1)
    if ivmax is None:
        ivmax = np.quantile(volume.flatten(),0.9)
        
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0]//2
    ax.axis(iaxis)

    ax.imshow(volume[ax.index],cmap=icmap,vmin = np.minimum(ivmin,ivmax), vmax = np.maximum(ivmax,ivmin))

    def update_slice(event):
        if event.key == 'j':
            ax.index = (ax.index - 1) % volume.shape[0]
        elif event.key == 'k':
            ax.index = (ax.index + 1) % volume.shape[0]
        ax.imshow(volume[ax.index],cmap=icmap,vmin=ivmin,vmax=ivmax)
        ax.axis(iaxis)
        fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', update_slice)
    plt.show()

def single_slice_viewer(I:np.array,ivmin:np.float64 = None,ivmax:np.float64 = None,icmap='gray',iaxis='off'):
    if ivmin is None:
        ivmin = np.quantile(I.flatten(),0.1)
    if ivmax is None:
        ivmax = np.quantile(I.flatten(),0.9)
           
    plt.figure()
    plt.imshow(I,cmap = icmap, vmin = np.minimum(ivmin,ivmax), vmax = np.maximum(ivmax,ivmin))
    plt.axis(iaxis)
    plt.show()

'''============================================================================
Image Visualization
[H] - created by JZhang prior to 20260331
============================================================================'''
def multi_section_viewer(volume: np.array, ivmin: np.float64 = None, ivmax: np.float64 = None, icmap='gray', iaxis='off', isection='row'):
    if ivmin is None:
        ivmin = np.quantile(volume.flatten(), 0.1)
    if ivmax is None:
        ivmax = np.quantile(volume.flatten(), 0.9)

    fig, ax = plt.subplots()
    ax.volume = volume
    ax.isection = isection

    if isection == 'slice':
        ax.index = volume.shape[0] // 2
        img = volume[ax.index, :, :]
    elif isection == 'row':
        ax.index = volume.shape[1] // 2
        img = volume[:, ax.index, :]
    elif isection == 'col':
        ax.index = volume.shape[2] // 2
        img = volume[:, :, ax.index]
    else:
        raise ValueError("isection must be 'slice', 'row', or 'col'")

    ax.axis(iaxis)
    im = ax.imshow(img, cmap=icmap,
                   vmin=np.minimum(ivmin, ivmax),
                   vmax=np.maximum(ivmin, ivmax))

    def update_slice(event):
        if ax.isection == 'slice':
            nsec = volume.shape[0]
        elif ax.isection == 'row':
            nsec = volume.shape[1]
        elif ax.isection == 'col':
            nsec = volume.shape[2]

        if event.key == 'j':
            ax.index = (ax.index - 1) % nsec
        elif event.key == 'k':
            ax.index = (ax.index + 1) % nsec
        else:
            return

        if ax.isection == 'slice':
            img = volume[ax.index, :, :]
        elif ax.isection == 'row':
            img = volume[:, ax.index, :]
        elif ax.isection == 'col':
            img = volume[:, :, ax.index]

        im.set_array(img)
        ax.set_title(f'{ax.isection}: {ax.index}')
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', update_slice)
    plt.show()
    
    
# Display the 3D volume
# plt.figure()
# for k in range(nslice): 
#     I = np.squeeze(vol.volume[k,:,:])
#     I = np.flipud(I)   
#     fig = plt.imshow(I,cmap='gray',vmin= 1.0, vmax=1.1)
#     plt.title('FORBILD Head in (x,y) Plane')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.colorbar()
#     plt.axis('off')
#     plt.show()