from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
from math import sin
from math import cos
import numpy as np
from scipy import linalg

""" This program reads a 3D point cloud data of a teapot
and attempts to reconstruct it using structure from motion
by tomasi kanadi factorization
"""

# specify number of frames
FRAMES = 36

# This function reads the 3D data point cloud data of teapot.mat
def read_3D():
    x = loadmat('teapot.mat')
    a = []
    b = []
    c = []
    for points in x['verts']:
        a.append(points[0])
        b.append(points[1])
        c.append(points[2])
    # plot 3D graph of data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a,b,c, c='b', marker='*')
    return a, b, c

# This function creates the 2D dataset for the teapot using orthographic projection camera
def create_2D_Dataset(x_points, y_points, z_points):
    y_angle = 0
    y_scale = 0
    # no of points in one frame
    length = len(x_points)
    # no of frames
    frames = FRAMES
    # for 'length' number of x points in 'frames'
    x_combined = np.zeros((frames, length))
    # for 'length' number of y points in 'frames'
    y_combined = np.zeros((frames, length))
    for j in range(frames):
        x_cal = []
        y_cal = []
        for i in range(length):
            # create array of x, y and z points 4x1
            vector = np.array([x_points[i], y_points[i], z_points[i], 1]).reshape(4, 1)
            # calculate 2D points from 3D points
            x_y_point = calculate_points(0, y_angle, 0, 0, y_scale, 0, vector)
            x_euclidean = x_y_point[0] 
            y_euclidean = x_y_point[1] 
            # get x and y coordinates of current frame
            x_cal.append(x_euclidean)
            y_cal.append(y_euclidean)
        # show the current frame's 2D dataset
        show_graph(x_cal, y_cal)
        # increment angle
        y_angle = y_angle + 5
        # increment y_translation
        y_scale = y_scale + 0
        # get all the x and y coordinates of all the frames
        x_combined[j] = np.array(x_cal, dtype = np.float32).reshape((x_combined[j].shape[0], ))
        y_combined[j] = np.array(y_cal, dtype = np.float32).reshape((y_combined[j].shape[0], ))
    return x_combined, y_combined


# This function calculates 2d points from 3d points
def calculate_points(alpha, beta, gamma, x, y, z, vector_points):
    # get rotation matrix
    matrix_R = rotation_matrix(alpha, beta, gamma)
    # get translation matrix
    vector_T = translation_vector(x, y, z)
    # calculate projection matrix
    matrix_P = projection_matrix(matrix_R, vector_T)
    # R * X 4x4 with 4x1
    points = np.matmul(matrix_P, vector_points)
    return points


# This function is to calculate the rotation matrix
def rotation_matrix(alpha, beta, gamma):
    # alpha for x beta for y and gamma for z
    r11 = cos(beta) * cos(gamma)
    r12 = (sin(alpha) * sin(beta) * cos(gamma)) - (cos(alpha) * sin(gamma))
    r13 = cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma)
    r21 = cos(beta) * sin(gamma)
    r22 = sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma)
    r23 = cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma)
    r31 = -sin(beta)
    r32 = cos(beta) * sin(alpha)
    r33 = cos(beta) * cos(alpha)
    # 4x3
    matrix_R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33], [0, 0, 0]])
    return matrix_R


# This function is to specify the translation vector 
def translation_vector(x, y, z):
    # 1x4
    vector_T = np.array([x, y, z, 1])
    return vector_T


# This function is to calculate the orthographic projection matrix
def projection_matrix(matrix_R, vector_T):
    matrix_R = np.append(matrix_R, vector_T.reshape(4, 1), axis=1)
    return matrix_R

    
# This function is to display  the 2D scatterplot graphs of 2D dataset generated
def show_graph(x, y):
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    scat = ax.scatter(x, y, color='r')
    plt.show()
    return

#This function generates measurements matrix W 
def generate_W_matrix(x_points, y_points):
    # number of frames and points
    frames = FRAMES
    length = x_points.shape[1]
    matrix_W = np.zeros((2 * frames, length))
    index = 0
    for i in range(frames):
        # set x row
        matrix_W[index] = x_points[i]
        # set y row
        matrix_W[index + 1] = y_points[i]
        index = index + 2
    return matrix_W


# This function executes the tomasi kanadi factorization using SVD
def tomasi_kanadi_factorization(matrix_W):
    # SVD transformation
    U, D, Vt = linalg.svd(matrix_W)
    # get rows X 3 block from U
    U_hat = U[: ,0:3]
    d1 = D[0]
    d2 = D[1]
    d3 = D[2]
    # get diagonal matrix for first three eigenvalues
    D_hat = np.array([[d1, 0, 0], [0, d2, 0], [0, 0, d3]])
    V = np.transpose(Vt)
    # get rows x 3 blocks from V transpose 
    V_hat = Vt[0:3 ,:]
    # now remove affine ambiguity
    # D = AQQinvX = (AQ) * (QinvX) where A = motion matrix and X=shape matrix
    R_hat = U_hat
    S_hat = V_hat
    # for AI = B specify the A and B matrix
    matrix_A = np.zeros((FRAMES * 3, 6))
    matrix_B = np.zeros((FRAMES * 3,1))
    """ 
    The source of these formulae is mentioned below
    source: http://cg.elte.hu/~hajder/vision/slides/lec03_multicamera.pdf
    """
    
    for i in range(FRAMES):
        # get 6 elements of R_hat for frame i R = 2FX3 2 rows for single R vector => 2x3
        mi1_x = R_hat[2*i,0]
        mi1_y = R_hat[2*i,1]
        mi1_z = R_hat[2*i,2]
        mi2_x = R_hat[2*i+1,0]
        mi2_y = R_hat[2*i+1,1]
        mi2_z = R_hat[2*i+1,2]
        # calculate matrix_A vector for 3 * i index
        matrix_A[3*i,0] = mi1_x*mi1_x
        matrix_A[3*i,1] = 2*mi1_x*mi1_y
        matrix_A[3*i,2] = 2*mi1_x*mi1_z
        matrix_A[3*i,3] = mi1_y*mi1_y
        matrix_A[3*i,4] = 2*mi1_y*mi1_z
        matrix_A[3*i,5] = mi1_z*mi1_z
         # calculate matrix_A vector for 3 * i + 1 index
        matrix_A[3*i+1,0] = mi2_x*mi2_x
        matrix_A[3*i+1,1] = 2*mi2_x*mi2_y
        matrix_A[3*i+1,2] = 2*mi2_x*mi2_z
        matrix_A[3*i+1,3] = mi2_y*mi2_y
        matrix_A[3*i+1,4] = 2*mi2_y*mi2_z
        matrix_A[3*i+1,5] = mi2_z*mi2_z
         # calculate matrix_A vector for 3 * i + 2 index
        matrix_A[3*i+2,0] = mi1_x*mi2_x
        matrix_A[3*i+2,1] = mi1_x*mi2_y + mi2_x*mi1_y
        matrix_A[3*i+2,2] = mi1_x*mi2_z + mi2_x*mi2_z
        matrix_A[3*i+2,3] = mi1_y*mi2_y
        matrix_A[3*i+2,4] = mi1_y*mi2_z + mi2_y*mi1_z
        matrix_A[3*i+2,5] = mi1_z + mi2_z
        # matrix_B for orthonormal vectors
        matrix_B[3*i] = 1
        matrix_B[3*i+1] = 1
        matrix_B[3*i+2] = 0
    
    # solve for I in the equation AI = B 
    # since matrix_A is not square so solve it using least squares method
    I = np.linalg.lstsq(matrix_A, matrix_B, rcond=None)[0]
    I = I.reshape(1, I.shape[0]*I.shape[1])[0]
    # L = Q*Q_transpose
    L = [[I[0], I[1], I[2]], [I[1], I[3], I[4]],[I[2], I[4], I[5]]]
    L = np.array(L)
    # apply cholesky factorization to get Q from Q*Q_transpose
    Q = np.linalg.cholesky(L)
    Q = np.transpose(Q)
    # motion matrix = R_hat * Q
    matrix_M = np.matmul(R_hat, Q)
    Qinv = np.linalg.inv(Q)
    # shape matrix = O_inverse, S_hat
    matrix_S = np.matmul(Qinv, S_hat)
    matrix_S = np.transpose(matrix_S)
    return matrix_M, matrix_S

# This function is to re project the caclulated shape
def re_project(matrix_S):
    x_cal = matrix_S[:,0]
    y_cal = matrix_S[:,1]
    z_cal = matrix_S[:,2]
    # plot the re caclulated shape points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(matrix_S[:,0],matrix_S[:,1],matrix_S[:,2], c='b', marker='*')
    return x_cal, y_cal, z_cal


# This function is to calculate the error from the original and calculated points
def calculate_error(x_original, y_original, z_original, x_cal, y_cal, z_cal):
    num_points = len(x_original)
    diff = 0
    error = 0
    for i in range(num_points):
        # euclidean distance
        diff = np.square(x_original[i] - x_cal[i]) + np.square(y_original[i] - y_cal[i]) + np.square(z_original[i] - z_cal[i])
        diff = np.sqrt(diff)
        # total error
        error = error + diff
    error = error / num_points
    return error

# main function 
def main():
    # read original 3D data
    x, y, z = read_3D()
    # create 2D dataset
    x_points, y_points = create_2D_Dataset(x, y, z)
    # calculate measurements matrix
    matrix_W = generate_W_matrix(x_points, y_points)
    #print(matrix_W.shape)
    # calculate motion and shape matrices
    matrix_M, matrix_S = tomasi_kanadi_factorization(matrix_W)
    #matrix_S = eliminate_affine_ambiguity(matrix_M, matrix_S, 20)
    # re project calculated shape
    x_cal, y_cal, z_cal = re_project(matrix_S)
    # calculate error
    error = calculate_error(x, y, z, x_cal, y_cal, z_cal)
    print(error)
    return

main()