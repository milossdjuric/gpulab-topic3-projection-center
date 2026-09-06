import argparse
import numpy as np
import math
import json
import h5py

def interpolate(proj,
                rotation_matrix,
                real_SDD,
                real_detector_width, 
                real_detector_height, 
                real_pixel_size,                                                      
                SDD,
                detector_width, 
                detector_height, 
                pixel_size):
    real_proj = np.zeros([real_detector_height, real_detector_width])

    for idx in range(real_proj.shape[1]):
        for idy in range(real_proj.shape[0]):
            if idx < real_proj.shape[1] and idy < real_proj.shape[0]:
                x = (idx - (real_detector_width - 1) / 2) * real_pixel_size
                y = ((real_detector_height - 1) / 2 - idy) * real_pixel_size
                z = -real_SDD

                nx = rotation_matrix[0][0] * x + rotation_matrix[0][1] * y + rotation_matrix[0][2] * z
                ny = rotation_matrix[1][0] * x + rotation_matrix[1][1] * y + rotation_matrix[1][2] * z
                nz = rotation_matrix[2][0] * x + rotation_matrix[2][1] * y + rotation_matrix[2][2] * z

                nx = nx / nz * (-SDD)
                ny = ny / nz * (-SDD)

                nidx = nx / pixel_size + (detector_width - 1) / 2
                nidy = -ny / pixel_size + (detector_height - 1) / 2

                if nidx >= 0 and nidx < detector_width - 1 and nidy >=0 and nidy < detector_height - 1:
                    idx_x = int(nidx)
                    idx_y = int(nidy) 
                    dx = nidx - idx_x
                    dy = nidy - idx_y

                    value = (1 - dx) * ((1 - dy) * proj[idx_y][idx_x] + dy * proj[idx_y + 1][idx_x]) + dx * ((1 - dy) * proj[idx_y][idx_x + 1] + dy * proj[idx_y + 1][idx_x + 1]) 
                    real_proj[idy][idx] = value
                else:
                    real_proj[idy][idx] = 0

    return real_proj



def get_real_projection(cb_para, real_cb_para, rotation_matrix, proj):
    SDD = cb_para['SDD']
    detector_width = int(cb_para['detector_width'])
    detector_height = int(cb_para['detector_height'])
    pixel_size = cb_para['pixel_size']

    real_SDD = real_cb_para['SDD']
    real_detector_width = int(real_cb_para['detector_width'])
    real_detector_height = int(real_cb_para['detector_height'])
    real_pixel_size = real_cb_para['pixel_size']
    
    

    real_proj = interpolate(proj,
                            rotation_matrix,
                            real_SDD,
                            real_detector_width, 
                            real_detector_height, 
                            real_pixel_size,                                                      
                            SDD,
                            detector_width, 
                            detector_height, 
                            pixel_size)

    return real_proj

def get_cb_para(cb_para, real_cb_pose, downsample_factor):
    real_cb_para = {}
    real_cb_para["num_projs"] = cb_para["num_projs"]
    real_cb_para["pixel_size"] = cb_para["pixel_size"] * downsample_factor
    real_cb_para["detector_width"] = cb_para["detector_width"] / downsample_factor
    real_cb_para["detector_height"] = cb_para["detector_height"] / downsample_factor

    SDD = cb_para["SDD"]
    SOD = cb_para["SOD"]
    x0 = real_cb_pose['center_point'][0]
    y0 = real_cb_pose['center_point'][1]
    xshift = real_cb_pose['xshift']
    alpha = real_cb_pose['alpha']
    beta = real_cb_pose['beta']

    sin_a = math.sin(alpha)
    cos_a = math.cos(alpha)
    sin_b = math.sin(beta)
    cos_b = math.cos(beta)

    real_SDD = np.linalg.norm([x0, y0, SDD])

    A = np.array([xshift, 0, -SOD])
    v_real = np.array([-sin_a, cos_a * cos_b, cos_a * sin_b])
    cross = np.cross(A, v_real)
    real_SOD = np.linalg.norm(cross) / np.linalg.norm(v_real)

    real_cb_para["SDD"] = real_SDD
    real_cb_para["SOD"] = real_SOD
    
    return real_cb_para

def get_rotation_matrix(cb_para, real_cb_pose):
    SDD = cb_para["SDD"]
    x0 = real_cb_pose['center_point'][0]
    y0 = real_cb_pose['center_point'][1]
    alpha = real_cb_pose['alpha']
    beta = real_cb_pose['beta']
    
    sin_a = math.sin(alpha)
    cos_a = math.cos(alpha)
    sin_b = math.sin(beta)
    cos_b = math.cos(beta)

    y_axis = np.array([-sin_a, cos_a * cos_b, cos_a * sin_b])
    z_axis = -np.array([x0, y0, -SDD])

    z_axis = z_axis / np.linalg.norm(z_axis)

    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    z = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])

    return R



def resampling(cb_para, projs, downsample, real_cb_pose):
    real_cb_para = get_cb_para(cb_para, real_cb_pose, downsample)
    rotation_matrix = get_rotation_matrix(cb_para, real_cb_pose)
    
    real_projs = np.zeros_like(projs)

    for i_proj in range(projs.shape[0]):
        proj = projs[i_proj]

        real_projs[i_proj] = get_real_projection(cb_para, real_cb_para, rotation_matrix, proj)

        print("Has resampled ", i_proj + 1, " projections")


    with h5py.File ('/lgrp/edu-2026-1-gpulab/projs_resample.hdf5', 'w') as file:
        file.create_dataset ('pixelSize', dtype = np.float64, data = real_cb_para["pixel_size"])
        file.create_dataset ('SDD', dtype = np.float64, data = real_cb_para["SDD"])
        file.create_dataset ('SOD', dtype = np.float64, data = real_cb_para["SOD"])
        file.create_dataset ('voxelSize', dtype = np.float64, data = cb_para["voxelSize"])
        file.create_dataset ('Volumen_num_xz', dtype = np.float64, data = cb_para["Volumen_num_xz"])
        file.create_dataset ('Volumen_num_y', dtype = np.float64, data = cb_para["Volumen_num_y"])
        file.create_dataset ('num_projs', dtype = np.float64, data = real_cb_para["num_projs"])
        file.create_dataset ('detector_width', dtype = np.float64, data = real_cb_para["detector_width"])
        file.create_dataset ('detector_height', dtype = np.float64, data = real_cb_para["detector_height"])
        file.create_dataset ('Angle', dtype = np.float64, data = cb_para["angles"])
        projection= file.create_dataset ('Projection', dtype = np.float32, shape = (real_cb_para["num_projs"],real_cb_para["detector_height"],real_cb_para["detector_width"]))   
                                                                  
        projection[:,:,:]=real_projs


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default='/lgrp/edu-2026-1-gpulab/projs_change.hdf5', type=str, help="Path to input data.")
    parser.add_argument("--downsample", default=1, help="Downsampling factor")

    args = parser.parse_args()
    input_data_path = args.data
    downsample_factor = args.downsample

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
        
    #use the results from searching, also can use '/lgrp/edu-2026-1-gpulab/reference_cb_pose.json'
    with open('./real_cb_pose.json', 'r') as f:
        real_cb_pose = json.load(f)

    resampling(cb_para, projs, downsample_factor, real_cb_pose)
