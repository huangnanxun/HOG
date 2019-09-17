#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

        
def get_differential_filter():
    filter_x = [[1,0,-1],[1,0,-1],[1,0,-1]]
    filter_y = [[1,1,1],[0,0,0],[-1,-1,-1]]
    return filter_x, filter_y

def filter_image(im, filter):
    im_f = np.zeros((np.size(im,0),np.size(im,1)));
    #suppose the filter is N by N
    padding_len = int(np.ceil((np.size(filter,0)-1)/2));
    im = np.pad(im,((padding_len,padding_len),(padding_len,padding_len)),'constant');
    center_k = np.floor(np.size(filter,0)/2);
    center_l = np.floor(np.size(filter,1)/2);
    x_range_hund = int(np.floor((np.size(im,0)-1)/100))
    if (filter == get_differential_filter()[0]):
        fil_type = 'filter_x'
    else:
        fil_type = 'filter_y'
    for i in range(1,np.size(im,0)-1):
        #Here is a timer to ensure the program is still running
        if (i % x_range_hund == 0):
            k = i // x_range_hund
            fin_k = k//2
            rem_k = 50 - fin_k
            print("\r","|"*fin_k+"."*rem_k,"Finish filter image with %s %i%% " %(fil_type,k), end="")
        for j in range(1,np.size(im,1)-1):
            v = 0;
            break_flag = False;
            for k in range(0,np.size(filter,0)):
                if(break_flag == True):
                        break;
                for l in range(0,np.size(filter,1)):
                    i1 = int(i + k - center_k);
                    j1 = int(j + l - center_l);
                    if (i1 < 0 or i1 > np.size(im,0) or j1 < 0 or j1 > np.size(im,1)):
                        break_flag = True;
                        break;
                    else:
                        v = v + (im[i1][j1])*(filter[k][l]);
            im_f[i-1][j-1] = v;
    im_filtered = im_f
    print("\n")
    return im_filtered


def get_gradient(im_dx, im_dy):
    grad_mag = np.zeros((np.size(im_dx,0),np.size(im_dx,1)));
    grad_angle = np.zeros((np.size(im_dx,0),np.size(im_dx,1)));
    for i in range(0,np.size(im_dx,0)):
        for j in range(0,np.size(im_dx,1)):
            grad_mag[i][j] = np.sqrt(im_dx[i][j]**2+im_dy[i][j]**2)
            if(grad_mag[i][j] == 0):
                grad_angle[i][j] = 0
            else:
                grad_angle[i][j] = np.arctan(im_dy[i][j]/im_dx[i][j])
            if (grad_angle[i][j] < 0):
                grad_angle[i][j] = grad_angle[i][j] + np.pi
    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    M = int(np.floor(np.size(grad_mag,0)/cell_size));
    N = int(np.floor(np.size(grad_mag,1)/cell_size));
    ori_histo = np.zeros((M,N,6));
    for i in range(0,M):
        for j in range(0,N):
            bin1 = 0
            bin2 = 0
            bin3 = 0
            bin4 = 0
            bin5 = 0
            bin6 = 0
            for p in range(0,cell_size):
                for q in range(0,cell_size):
                    detect_angle = grad_angle[i*cell_size+p][j*cell_size+q]
                    detect_mag = grad_mag[i*cell_size+p][j*cell_size+q]
                    if (detect_angle >= 11/12*np.pi or detect_angle < 1/12*np.pi):
                        bin1 = bin1 + detect_mag
                    if (detect_angle >= 1/12*np.pi and detect_angle < 3/12*np.pi):
                        bin2 = bin2 + detect_mag
                    if (detect_angle >= 3/12*np.pi and detect_angle < 5/12*np.pi):
                        bin3 = bin3 + detect_mag
                    if (detect_angle >= 5/12*np.pi and detect_angle < 7/12*np.pi):
                        bin4 = bin4 + detect_mag
                    if (detect_angle >= 7/12*np.pi and detect_angle < 9/12*np.pi):
                        bin5 = bin5 + detect_mag
                    if (detect_angle >= 9/12*np.pi and detect_angle < 11/12*np.pi):
                        bin6 = bin6 + detect_mag
            ori_histo[i][j][0] = bin1;
            ori_histo[i][j][1] = bin2;
            ori_histo[i][j][2] = bin3;
            ori_histo[i][j][3] = bin4;
            ori_histo[i][j][4] = bin5;
            ori_histo[i][j][5] = bin6;
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    M = np.size(ori_histo,0)
    N = np.size(ori_histo,1)
    M_hist = M-(block_size-1)
    N_hist = N-(block_size-1)
    ori_histo_normalized = np.zeros((M_hist,N_hist,6*(block_size**2)));
    constant_e = 0.001
    for i in range(0,M_hist):
        for j in range(0,N_hist):
            w_ori = 0
            for a in range(0,block_size):
                for b in range(0,block_size):
                    w_ori = w_ori + ori_histo[i+a][j+b][0]**2+ori_histo[i+a][j+b][1]**2+ori_histo[i+a][j+b][2]**2+ori_histo[i+a][j+b][3]**2+ori_histo[i+a][j+b][4]**2+ori_histo[i+a][j+b][5]**2
            w_ori = np.sqrt(w_ori + constant_e**2)
            for a in range(0,block_size):
                for b in range(0,block_size):
                    ori_histo_normalized[i][j][6*(a*block_size+b)] = ori_histo[i+a][j+b][0]/w_ori
                    ori_histo_normalized[i][j][6*(a*block_size+b)+1] = ori_histo[i+a][j+b][1]/w_ori
                    ori_histo_normalized[i][j][6*(a*block_size+b)+2] = ori_histo[i+a][j+b][2]/w_ori
                    ori_histo_normalized[i][j][6*(a*block_size+b)+3] = ori_histo[i+a][j+b][3]/w_ori
                    ori_histo_normalized[i][j][6*(a*block_size+b)+4] = ori_histo[i+a][j+b][4]/w_ori
                    ori_histo_normalized[i][j][6*(a*block_size+b)+5] = ori_histo[i+a][j+b][5]/w_ori
    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    x_dif_img = filter_image(im,get_differential_filter()[0])
    y_dif_img = filter_image(im,get_differential_filter()[1])
    grad_info = get_gradient(x_dif_img,y_dif_img)
    ori_histo_info = build_histogram(grad_info[0], grad_info[1], 8)
    hog = get_block_descriptor(ori_histo_info, 2)
    # visualize to verify
    visualize_hog(im, hog, 8, 2)
    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    #fig = plt.figure(figsize=(16,16)) <- I get this line from my fellow to expand the image while testing
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


if __name__=='__main__':
    im = cv2.imread('cameraman.tif', 0)
    #im = cv2.imread('apple_86.jpeg', 0)
    hog = extract_hog(im)







