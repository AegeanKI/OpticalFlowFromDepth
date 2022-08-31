#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>

extern "C" {
    __global__ void multiply_them(float *dest, float *a, float *b) {
        printf("Hello");
        //const int i = threadIdx.x;
        // const int i =  blockIdx.x * blockDim.x + threadIdx.x;
        const int i = threadIdx.y * blockDim.x + threadIdx.x;
        dest[i] = a[i] * b[i];
    }

    __global__ void forward_warping_flow(const double *obj, const double *safe_y, const double *safe_x, const double *depth,
                                         int dim_b, int dim_c, int dim_h, int dim_w, double same_range, double *output,
                                         double *dlut, double *dlut_count) {
        //                 dim_b, dim_c, dim_h, dim_w
        auto c_idx = [&](int b, int z, int y, int x) {
            return b * dim_c * dim_h * dim_w + z * dim_h * dim_w + y * dim_w + x; 
        };
        //               dim_b,     1  dim_h  dim_w
        auto idx = [&](int b, int z, int y, int x) {
            return b * 1 * dim_h * dim_w + z * dim_h * dim_w + y * dim_w + x;
        };

        for (int k = 0; k < dim_b; k++) {
            for (int i = 0; i < dim_h; i++) {
                for (int j = 0; j < dim_w; j++) {
                    double cur_depth = depth[idx(k, 0, i, j)];
                    double x = safe_x[idx(k, 0, i, j)];
                    double y = safe_y[idx(k, 0, i, j)];

                    int x_f = (int)floor(x);
                    int y_f = (int)floor(y);
                    int x_c = x_f + 1;
                    int y_c = y_f + 1;

                    if(x_f >= 0 && x_c < dim_w && y_f >= 0 && y_c < dim_h){
                        int nw_depth_idx = idx(k, 0, y_f, x_f);
                        int ne_depth_idx = idx(k, 0, y_f, x_c);
                        int sw_depth_idx = idx(k, 0, y_c, x_f);
                        int se_depth_idx = idx(k, 0, y_c, x_c);

                        double nw_depth = dlut[nw_depth_idx] / (double) dlut_count[nw_depth_idx];
                        double ne_depth = dlut[ne_depth_idx] / (double) dlut_count[ne_depth_idx];
                        double sw_depth = dlut[sw_depth_idx] / (double) dlut_count[sw_depth_idx];
                        double se_depth = dlut[se_depth_idx] / (double) dlut_count[se_depth_idx];

                        if (nw_depth == 100.0 || nw_depth > cur_depth + same_range) {
                            for (int c = 0; c < dim_c; c++) {
                                int obj_idx = c_idx(k, c, i, j);
                                int nw_flow_idx = c_idx(k, c, y_f, x_f);
                                output[nw_flow_idx] = obj[obj_idx];
                            }
                            dlut[nw_depth_idx] = cur_depth;
                            dlut_count[nw_depth_idx] = 1;
                        } else if (std::abs(nw_depth - cur_depth) <= same_range) {
                            for (int c = 0; c < dim_c; c++) {
                                int obj_idx = c_idx(k, c, i, j);
                                int nw_flow_idx = c_idx(k, c, y_f, x_f);
                                output[nw_flow_idx] += obj[obj_idx];
                            }
                            dlut[nw_depth_idx] += cur_depth;
                            dlut_count[nw_depth_idx]++;
                        }
                        if (ne_depth == 100.0 || ne_depth > cur_depth + same_range) {
                            for (int c = 0; c < dim_c; c++) {
                                int obj_idx = c_idx(k, c, i, j);
                                int ne_flow_idx = c_idx(k, c, y_f, x_c);
                                output[ne_flow_idx] = obj[obj_idx];
                            }
                            dlut[ne_depth_idx] = cur_depth;
                            dlut_count[ne_depth_idx] = 1;
                        } else if (std::abs(ne_depth - cur_depth) <= same_range) {
                            for (int c = 0; c < dim_c; c++) {
                                int obj_idx = c_idx(k, c, i, j);
                                int ne_flow_idx = c_idx(k, c, y_f, x_c);
                                output[ne_flow_idx] += obj[obj_idx];
                            }
                            dlut[ne_depth_idx] += cur_depth;
                            dlut_count[ne_depth_idx]++;
                        }
                        if (sw_depth == 100.0 || sw_depth > cur_depth + same_range) {
                            for (int c = 0; c < dim_c; c++) {
                                int obj_idx = c_idx(k, c, i, j);
                                int sw_flow_idx = c_idx(k, c, y_c, x_f);
                                output[sw_flow_idx] = obj[obj_idx];
                            }
                            dlut[sw_depth_idx] = cur_depth;
                            dlut_count[sw_depth_idx] = 1;
                        } else if (std::abs(sw_depth - cur_depth) <= same_range) {
                            for (int c = 0; c < dim_c; c++) {
                                int obj_idx = c_idx(k, c, i, j);
                                int sw_flow_idx = c_idx(k, c, y_c, x_f);
                                output[sw_flow_idx] += obj[obj_idx];
                            }
                            dlut[sw_depth_idx] += cur_depth;
                            dlut_count[sw_depth_idx]++;
                        }
                        if (se_depth == 100.0 || se_depth > cur_depth + same_range) {
                            for (int c = 0; c < dim_c; c++) {
                                int obj_idx = c_idx(k, c, i, j);
                                int se_flow_idx = c_idx(k, c, y_c, x_c);
                                output[se_flow_idx] = obj[obj_idx];
                            }
                            dlut[se_depth_idx] = cur_depth;
                            dlut_count[se_depth_idx] = 1;
                        } else if (std::abs(se_depth - cur_depth) <= same_range) {
                            for (int c = 0; c < dim_c; c++) {
                                int obj_idx = c_idx(k, c, i, j);
                                int se_flow_idx = c_idx(k, c, y_c, x_c);
                                output[se_flow_idx] += obj[obj_idx];
                            }
                            dlut[se_depth_idx] += cur_depth;
                            dlut_count[se_depth_idx]++;
                        }
                    }
                }
            }
        }
        for (int k = 0; k < dim_b; k++) {
            for (int i = 0; i < dim_h; i++) {
                for (int j = 0; j < dim_w; j++) {
                    int depth_idx = idx(k, 0, i, j);
                    if (!dlut_count[depth_idx]) continue;

                    for (int c = 0; c < dim_c; c++) {
                        int obj_idx = c_idx(k, c, i, j);
                        output[obj_idx] = output[obj_idx] / (double) dlut_count[depth_idx];
                    }
                }
            }
        }
        return;
    }
}
