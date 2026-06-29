import argparse
import math
import numpy as np
import h5py
import json

def get_linear_interpolate_MSE(N,
                               sino_input, 
                               SDD, 
                               detector_width, 
                               detector_height, 
                               pixel_size, 
                               alpha, 
                               beta, 
                               theta_0, 
                               nearest_theta):
    pixel_MSE = np.zeros((N), dtype=np.float64)

    for idx in range(nearest_theta.shape[0]):
        if idx < nearest_theta.shape[0]:
            theta = nearest_theta[idx] + theta_0
            reflect_theta = -nearest_theta[idx] + theta_0
            tan_t = math.tan(theta)
            tan_rt = math.tan(reflect_theta)

            sin_a = math.sin(alpha)
            cos_a = math.cos(alpha)
            sin_b = math.sin(beta)
            cos_b = math.cos(beta)

            temp = SDD / (tan_t * sin_a * sin_b + cos_b)
            rtemp = SDD / (tan_rt * sin_a * sin_b + cos_b)

            y = temp * (-tan_t * sin_a * cos_b + sin_b)
            ry = rtemp * (-tan_rt * sin_a * cos_b + sin_b)

            x = -temp * tan_t * cos_a
            rx = -rtemp * tan_rt * cos_a

            y = (detector_height - 1) / 2 - y / pixel_size
            ry = (detector_height - 1) / 2 - ry / pixel_size
            x = x / pixel_size + (detector_width - 1) / 2
            rx = rx / pixel_size + (detector_width - 1) / 2

            if x >= 0 and x < detector_width - 1 and y >=0 and y < detector_height - 1:
                idx_x = int(x)
                idx_y = int(y) 
                dx = x - idx_x
                dy = y - idx_y

                f_theta = (1 - dx) * ((1 - dy) * sino_input[idx_y][idx_x] + dy * sino_input[idx_y + 1][idx_x]) + dx * ((1 - dy) * sino_input[idx_y][idx_x + 1] + dy * sino_input[idx_y + 1][idx_x + 1])   

            if rx >= 0 and rx < detector_width - 1 and ry >=0 and ry < detector_height - 1:
                idx_x = int(rx)
                idx_y = int(ry)
                dx = rx - idx_x
                dy = ry - idx_y

                f_rtheta = (1 - dx) * ((1 - dy) * sino_input[idx_y][idx_x] + dy * sino_input[idx_y + 1][idx_x]) + dx * ((1 - dy) * sino_input[idx_y][idx_x + 1] + dy * sino_input[idx_y + 1][idx_x + 1]) 
            pixel_MSE[idx] = (f_theta - f_rtheta) * (f_theta - f_rtheta)   

    return pixel_MSE                

def find_conebeam_COR_line_forward(cb_para, sino_input, xshift = 0.0, alpha = 0.0, beta = 0.0, range=30):
    SOD = cb_para['SOD']
    SDD = cb_para['SDD']
    detector_width = int(cb_para['detector_width'])
    detector_height = int(cb_para['detector_height'])
    pixel_size = cb_para['pixel_size']

    N = 1000
    
    sin_a = math.sin(alpha)
    cos_a = math.cos(alpha)
    sin_b = math.sin(beta)
    cos_b = math.cos(beta)

    rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a * cos_b, cos_a * cos_b, -sin_b], [sin_a * sin_b, cos_a * sin_b, cos_b]])

    b_vector = SDD * cos_a * np.array([sin_b, xshift * cos_b])
    det_A = -sin_a * (-xshift * cos_a * sin_b + SOD * sin_a) - cos_a * cos_b * SOD * cos_a * cos_b
    inverse_A = np.array([[(-xshift * cos_a * sin_b + SOD * sin_a), -cos_a * cos_b], [-SOD * cos_a * cos_b, -sin_a]])
    if det_A != 0:
        inverse_A = inverse_A / det_A
    x_0 = inverse_A[0][0] * b_vector[0] + inverse_A[0][1] * b_vector[1]
    y_0 = inverse_A[1][0] * b_vector[0] + inverse_A[1][1] * b_vector[1]

    x_p = rotation_matrix[0][0] * x_0 + rotation_matrix[1][0] * y_0 - rotation_matrix[2][0] * SDD
    y_p = rotation_matrix[0][1] * x_0 + rotation_matrix[1][1] * y_0 - rotation_matrix[2][1] * SDD   
    z_p = rotation_matrix[0][2] * x_0 + rotation_matrix[1][2] * y_0 - rotation_matrix[2][2] * SDD
    
    theta_0 = math.atan(x_p / z_p)

    nearest_theta = np.arange(N) * range / 1000 / 180 * np.pi

    pixel_MSE = np.zeros((N), dtype=np.float64)

    pixel_MSE = get_linear_interpolate_MSE(N,
                                           sino_input, 
                                           SDD, 
                                           detector_width, 
                                           detector_height, 
                                           pixel_size, 
                                           alpha, 
                                           beta, 
                                           theta_0, 
                                           nearest_theta)
    
    MSE = pixel_MSE.sum(axis = 0) / N

    return MSE, x_0, y_0

def Compute_COR(cb_para, projs, args):
    xshift = args.xshift / 1000
    alpha = args.alpha / 180 * np.pi
    beta = args.beta / 180 * np.pi
    xshift_step = args.xshift_step / 1000
    alpha_step = args.alpha_step / 180 * np.pi
    beta_step = args.beta_step / 180 * np.pi

    sinogram = projs.sum(axis=0)
    sinogram = (sinogram - np.min(sinogram)) / (np.max(sinogram) - np.min(sinogram))

    xshift_vals = np.arange(-xshift, xshift + xshift_step / 2, xshift_step)
    alpha_vals = np.arange(-alpha, alpha + alpha_step / 2, alpha_step)
    beta_vals = np.arange(-beta, beta + beta_step / 2, beta_step)

    X, A, B = np.meshgrid(
        xshift_vals,
        alpha_vals,
        beta_vals,
        indexing="ij"
    )

    params = np.stack([X, A, B, np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)], axis=-1).reshape(-1, 6)

    for i in range(params.shape[0]):
        params[i][3], params[i][4], params[i][5] = find_conebeam_COR_line_forward(cb_para, sinogram, params[i][0], params[i][1], params[i][2])
        print(i, params[i][3], params[i][0] * 1000, params[i][1] / np.pi * 180, params[i][2] / np.pi * 180)

    idx = np.argmin(params[:, 3])

    real_cb_pose = {}
    real_cb_pose['center_point'] = [params[idx][4], params[idx][5]]
    real_cb_pose['xshift'] = params[idx][0]
    real_cb_pose['alpha'] = params[idx][1]
    real_cb_pose['beta'] = params[idx][2]
    real_cb_pose['MSE'] = params[idx][3]

    print("MSE: ", params[idx][3])
    print("xshift: ", params[idx][0] * 1000)
    print("alpha: ", params[idx][1] / np.pi * 180)
    print("beta: ", params[idx][2] / np.pi * 180)

    return real_cb_pose

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default='/lgrp/edu-2026-1-gpulab/projs_change.hdf5', type=str, help="Path to input data.")
    parser.add_argument("--xshift", default=40.0, type=float, help="Range of xshift (mm).")
    parser.add_argument("--alpha", default=10.0, type=float, help="Range of alpha (degree).")
    parser.add_argument("--beta", default=10.0, type=float, help="Range of beta (degree).")
    parser.add_argument("--xshift_step", default=1.0, type=float, help="Step of xshift (mm).")
    parser.add_argument("--alpha_step", default=1.0, type=float, help="Step of alpha (degree).")
    parser.add_argument("--beta_step", default=1.0, type=float, help="Step of beta (degree).")

    args = parser.parse_args()
    input_data_path = args.data

    with h5py.File(input_data_path, 'r') as f:

        voxelSize=f['voxelSize'][()] 
        Volumen_num_xz=int(f['Volumen_num_xz'][()] )
        Volumen_num_y=int(f['Volumen_num_y'][()] )
        SDD = f['SDD'][()] 
        SOD =f['SOD'][()]
        magnification=SDD/SOD
        pixelSize =f['pixelSize'][()] 

        num_projs=int(f['num_projs'][()] )
        detector_width=int(f['detector_width'][()] )
        detector_height=int(f['detector_height'][()] )
        angles=f['Angle'][()]

        cb_para={
            'num_projs': num_projs,
            'SDD': SDD,
            'SOD': SOD,
            'pixel_size': pixelSize ,
            'voxelSize': voxelSize,
            'Volumen_num_xz': Volumen_num_xz,
            'Volumen_num_y': Volumen_num_y,
            'detector_width': detector_width,
            'detector_height': detector_height,
            'angles': angles,
            }
        
        projs = f['Projection'][()][:]

    real_cb_pose = Compute_COR(cb_para, projs, args)

    with open("./real_cb_pose.json", 'w') as f:
        json.dump(real_cb_pose, f)