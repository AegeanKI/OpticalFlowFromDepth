#include <stdlib.h>
#include <stdio.h>
// #include <iostream>
#include <algorithm>
#include <math.h>
#define valid(X, Y, W)  (Y*W*5+X*5+3)
#define collision(X, Y, W)  (Y*W*5+X*5+4)

extern "C" {
    void warp_img(const double *img, const double *safe_y, const double *safe_x, const double *depth,
                int dim_h, int dim_w, double same_range, double *output) {
        int dim_c = 3;
        auto dim3_idx = [&](int y, int x, int z) {
            return y * dim_c * dim_w + x * dim_c + z;
        };
        auto dim2_idx = [&](int y, int x) {
            return y * dim_w + x;
        };

        double *dlut = (double *)malloc(dim_h * dim_w * sizeof(double));
        std::fill(dlut, dlut + dim_h * dim_w, 100.0);
        bool *warped = (bool *)calloc(dim_h * dim_w, sizeof(bool));

        for (int i = 0; i < dim_h; i++) {
            for (int j = 0; j < dim_w; j++) {
                double x = safe_x[dim2_idx(i, j)];
                double y = safe_y[dim2_idx(i, j)];

                int x_f = (int)floor(x);
                int y_f = (int)floor(y);
                int x_c = x_f + 1;
                int y_c = y_f + 1;

                if (!warped[dim2_idx(y_f, x_f)] ||
                    depth[dim2_idx(i, j)] < dlut[dim2_idx(y_f, x_f)] - same_range) {
                    // nothing
                } else if ((x - (double)x_f > 0.5) &&
                    (!warped[dim2_idx(y_f, x_c)] || depth[dim2_idx(i, j)] < dlut[dim2_idx(y_f, x_c)] - same_range)) {
                    x++;
                } else if ((y - (double)y_f > 0.5) &&
                    (!warped[dim2_idx(y_c, x_f)] || depth[dim2_idx(i, j)] < dlut[dim2_idx(y_c, x_f)] - same_range)) {
                    y++;
                } else if ((x - (double)x_f > 0.5) && (y - (double)y_f > 0.5) &&
                    (!warped[dim2_idx(y_c, x_c)] || depth[dim2_idx(i, j)] < dlut[dim2_idx(y_c, x_c)] - same_range)) {
                    x++;
                    y++;
                }

                int x_new = (int)floor(x);
                int y_new = (int)floor(y);
        
                if (depth[dim2_idx(i, j)] < dlut[dim2_idx(y_new, x_new)] - same_range) {
                    for (int z = 0; z < dim_c; z++) {
                        output[dim3_idx(y_new, x_new, z)] = img[dim3_idx(i, j, z)];
                    }
                    dlut[dim2_idx(y_new, x_new)] = depth[dim2_idx(i, j)];
                    warped[dim2_idx(y_new, x_new)] = 1;
                }
            }
        }

        bool onepixel_blank = true;
        while (onepixel_blank) {
            onepixel_blank = false;
            for (int i = 1; i < dim_h - 1; i++) {
                for (int j = 1; j < dim_w - 1; j++) {
                    if (warped[dim2_idx(i, j)]) continue;

                    if (warped[dim2_idx(i, j - 1)] && warped[dim2_idx(i, j + 1)]) {
                        for (int z = 0; z < dim_c; z++) {
                            output[dim3_idx(i, j, z)] = (output[dim3_idx(i, j - 1, z)] + output[dim3_idx(i, j + 1, z)]) / (double)2.;
                        }
                        warped[dim2_idx(i, j)] = 1;
                        onepixel_blank = true;
                    } else if (warped[dim2_idx(i - 1, j)] && warped[dim2_idx(i + 1, j)]) {
                        for (int z = 0; z < dim_c; z++) {
                            output[dim3_idx(i, j, z)] = (output[dim3_idx(i - 1, j, z)] + output[dim3_idx(i + 1, j, z)]) / (double)2.;
                        }
                        warped[dim2_idx(i, j)] = 1;
                        onepixel_blank = true;
                    } else if (warped[dim2_idx(i - 1, j - 1)] && warped[dim2_idx(i + 1, j + 1)]) {
                        for (int z = 0; z < dim_c; z++) {
                            output[dim3_idx(i, j, z)] = (output[dim3_idx(i - 1, j - 1, z)] + output[dim3_idx(i + 1, j + 1, z)]) / (double)2.;
                        }
                        warped[dim2_idx(i, j)] = 1;
                        onepixel_blank = true;
                    } else if (warped[dim2_idx(i + 1, j - 1)] && warped[dim2_idx(i - 1, j + 1)]) {
                        for (int z = 0; z < dim_c; z++) {
                            output[dim3_idx(i, j, z)] = (output[dim3_idx(i + 1, j - 1, z)] + output[dim3_idx(i - 1, j + 1, z)]) / (double)2.;
                        }
                        warped[dim2_idx(i, j)] = 1;
                        onepixel_blank = true;
                    }

                }
            }
        }

        free(dlut);
        free(warped);
        return;
    }

    void warp_depth(const double *safe_y, const double *safe_x, const double *depth,
                    int dim_h, int dim_w, double same_range, double *output) {
        auto dim2_idx = [&](int y, int x) {
            return y * dim_w + x;
        };

        double *dlut = (double *)malloc(dim_h * dim_w * sizeof(double));
        std::fill(dlut, dlut + dim_h * dim_w, 100.0);
        bool *warped = (bool *)calloc(dim_h * dim_w, sizeof(bool));

        for (int i = 0; i < dim_h; i++) {
            for (int j = 0; j < dim_w; j++) {
                double x = safe_x[dim2_idx(i, j)];
                double y = safe_y[dim2_idx(i, j)];

                int x_f = (int)floor(x);
                int y_f = (int)floor(y);
                int x_c = x_f + 1;
                int y_c = y_f + 1;

                if (!warped[dim2_idx(y_f, x_f)] ||
                    depth[dim2_idx(i, j)] < dlut[dim2_idx(y_f, x_f)] - same_range) {
                    // nothing
                } else if ((x - (double)x_f > 0.5) &&
                    (!warped[dim2_idx(y_f, x_c)] || depth[dim2_idx(i, j)] < dlut[dim2_idx(y_f, x_c)] - same_range)) {
                    x++;
                } else if ((y - (double)y_f > 0.5) &&
                    (!warped[dim2_idx(y_c, x_f)] || depth[dim2_idx(i, j)] < dlut[dim2_idx(y_c, x_f)] - same_range)) {
                    y++;
                } else if ((x - (double)x_f > 0.5) && (y - (double)y_f > 0.5) &&
                    (!warped[dim2_idx(y_c, x_c)] || depth[dim2_idx(i, j)] < dlut[dim2_idx(y_c, x_c)] - same_range)) {
                    x++;
                    y++;
                }

                int x_new = (int)floor(x);
                int y_new = (int)floor(y);
        
                if (depth[dim2_idx(i, j)] < dlut[dim2_idx(y_new, x_new)] - same_range) {
                    output[dim2_idx(y_new, x_new)] = depth[dim2_idx(i, j)];
                    dlut[dim2_idx(y_new, x_new)] = depth[dim2_idx(i, j)];
                    warped[dim2_idx(y_new, x_new)] = 1;
                }
            }
        }

        bool onepixel_blank = true;
        while (onepixel_blank) {
            onepixel_blank = false;
            for (int i = 1; i < dim_h - 1; i++) {
                for (int j = 1; j < dim_w - 1; j++) {
                    if (warped[dim2_idx(i, j)]) continue;

                    if (warped[dim2_idx(i, j - 1)] && warped[dim2_idx(i, j + 1)]) {
                        output[dim2_idx(i, j)] = (output[dim2_idx(i, j - 1)] + output[dim2_idx(i, j + 1)]) / (double)2.;
                        warped[dim2_idx(i, j)] = 1;
                        onepixel_blank = true;
                    } else if (warped[dim2_idx(i - 1, j)] && warped[dim2_idx(i + 1, j)]) {
                        output[dim2_idx(i, j)] = (output[dim2_idx(i - 1, j)] + output[dim2_idx(i + 1, j)]) / (double)2.;
                        warped[dim2_idx(i, j)] = 1;
                        onepixel_blank = true;
                    } else if (warped[dim2_idx(i - 1, j - 1)] && warped[dim2_idx(i + 1, j + 1)]) {
                        output[dim2_idx(i, j)] = (output[dim2_idx(i - 1, j - 1)] + output[dim2_idx(i + 1, j + 1)]) / (double)2.;
                        warped[dim2_idx(i, j)] = 1;
                        onepixel_blank = true;
                    } else if (warped[dim2_idx(i + 1, j - 1)] && warped[dim2_idx(i - 1, j + 1)]) {
                        output[dim2_idx(i, j)] = (output[dim2_idx(i + 1, j - 1)] + output[dim2_idx(i - 1, j + 1)]) / (double)2.;
                        warped[dim2_idx(i, j)] = 1;
                        onepixel_blank = true;
                    }

                }
            }
        }

        free(dlut);
        free(warped);
        return;
    }

    void warp_flow(const double *flow, const double *safe_y, const double *safe_x, const double *depth,
                   int dim_h, int dim_w, double same_range, double *output) {
        int dim_c = 2;
        auto dim3_idx = [&](int y, int x, int z) {
            return y * dim_c * dim_w + x * dim_c + z;
        };
        auto dim2_idx = [&](int y, int x) {
            return y * dim_w + x;
        };

        double *dlut = (double *)malloc(dim_h * dim_w * sizeof(double));
        std::fill(dlut, dlut + dim_h * dim_w, 100.0);
        bool *warped = (bool *)calloc(dim_h * dim_w, sizeof(bool));

        for (int i = 0; i < dim_h; i++) {
            for (int j = 0; j < dim_w; j++) {
                double x = safe_x[dim2_idx(i, j)];
                double y = safe_y[dim2_idx(i, j)];

                int x_f = (int)floor(x);
                int y_f = (int)floor(y);
                int x_c = x_f + 1;
                int y_c = y_f + 1;

                if (!warped[dim2_idx(y_f, x_f)] ||
                    depth[dim2_idx(i, j)] < dlut[dim2_idx(y_f, x_f)] - same_range) {
                    // nothing
                } else if ((x - (double)x_f > 0.5) &&
                    (!warped[dim2_idx(y_f, x_c)] || depth[dim2_idx(i, j)] < dlut[dim2_idx(y_f, x_c)] - same_range)) {
                    x++;
                } else if ((y - (double)y_f > 0.5) &&
                    (!warped[dim2_idx(y_c, x_f)] || depth[dim2_idx(i, j)] < dlut[dim2_idx(y_c, x_f)] - same_range)) {
                    y++;
                } else if ((x - (double)x_f > 0.5) && (y - (double)y_f > 0.5) &&
                    (!warped[dim2_idx(y_c, x_c)] || depth[dim2_idx(i, j)] < dlut[dim2_idx(y_c, x_c)] - same_range)) {
                    x++;
                    y++;
                }

                int x_new = (int)floor(x);
                int y_new = (int)floor(y);
        
                if (depth[dim2_idx(i, j)] < dlut[dim2_idx(y_new, x_new)] - same_range) {
                    for (int z = 0; z < dim_c; z++) {
                        output[dim3_idx(y_new, x_new, z)] = flow[dim3_idx(i, j, z)];
                    }
                    dlut[dim2_idx(y_new, x_new)] = depth[dim2_idx(i, j)];
                    warped[dim2_idx(y_new, x_new)] = 1;
                }
            }
        }

        bool onepixel_blank = true;
        while (onepixel_blank) {
            onepixel_blank = false;
            for (int i = 1; i < dim_h - 1; i++) {
                for (int j = 1; j < dim_w - 1; j++) {
                    if (warped[dim2_idx(i, j)]) continue;

                    if (warped[dim2_idx(i, j - 1)] && warped[dim2_idx(i, j + 1)]) {
                        for (int z = 0; z < dim_c; z++) {
                            output[dim3_idx(i, j, z)] = (output[dim3_idx(i, j - 1, z)] + output[dim3_idx(i, j + 1, z)]) / (double)2.;
                        }
                        warped[dim2_idx(i, j)] = 1;
                        onepixel_blank = true;
                    } else if (warped[dim2_idx(i - 1, j)] && warped[dim2_idx(i + 1, j)]) {
                        for (int z = 0; z < dim_c; z++) {
                            output[dim3_idx(i, j, z)] = (output[dim3_idx(i - 1, j, z)] + output[dim3_idx(i + 1, j, z)]) / (double)2.;
                        }
                        warped[dim2_idx(i, j)] = 1;
                        onepixel_blank = true;
                    } else if (warped[dim2_idx(i - 1, j - 1)] && warped[dim2_idx(i + 1, j + 1)]) {
                        for (int z = 0; z < dim_c; z++) {
                            output[dim3_idx(i, j, z)] = (output[dim3_idx(i - 1, j - 1, z)] + output[dim3_idx(i + 1, j + 1, z)]) / (double)2.;
                        }
                        warped[dim2_idx(i, j)] = 1;
                        onepixel_blank = true;
                    } else if (warped[dim2_idx(i + 1, j - 1)] && warped[dim2_idx(i - 1, j + 1)]) {
                        for (int z = 0; z < dim_c; z++) {
                            output[dim3_idx(i, j, z)] = (output[dim3_idx(i + 1, j - 1, z)] + output[dim3_idx(i - 1, j + 1, z)]) / (double)2.;
                        }
                        warped[dim2_idx(i, j)] = 1;
                        onepixel_blank = true;
                    }

                }
            }
        }

        free(dlut);
        free(warped);
        return;
    }

    void forward_warping_img(const double *img, const double *safe_y, const double *safe_x, const double *depth,
                             int dim_h, int dim_w, double same_range, double *output) {
        int dim_c = 3;
        auto dim3_idx = [&](int y, int x, int z) {
            return y * dim_c * dim_w + x * dim_c + z;
        };
        auto dim2_idx = [&](int y, int x) {
            return y * dim_w + x;
        };

        double *dlut = (double *)malloc(dim_h * dim_w * sizeof(double));
        std::fill(dlut, dlut + dim_h * dim_w, 100.0);
        int *dlut_count = (int *)calloc(dim_h * dim_w, sizeof(int));

        for (int i = 0; i < dim_h; i++) {
            for (int j = 0; j < dim_w; j++) {
                double cur_depth = depth[dim2_idx(i, j)];
                double x = safe_x[dim2_idx(i, j)];
                double y = safe_y[dim2_idx(i, j)];

                int x_f = (int)floor(x);
                int y_f = (int)floor(y);
                int x_c = x_f + 1;
                int y_c = y_f + 1;

                if(x_f >= 0 && x_c < dim_w && y_f >= 0 && y_c < dim_h){
                    int nw_depth_idx = dim2_idx(y_f, x_f);
                    int ne_depth_idx = dim2_idx(y_f, x_c);
                    int sw_depth_idx = dim2_idx(y_c, x_f);
                    int se_depth_idx = dim2_idx(y_c, x_c);

                    double nw_depth = dlut[nw_depth_idx] / (double) dlut_count[nw_depth_idx];
                    double ne_depth = dlut[ne_depth_idx] / (double) dlut_count[ne_depth_idx];
                    double sw_depth = dlut[sw_depth_idx] / (double) dlut_count[sw_depth_idx];
                    double se_depth = dlut[se_depth_idx] / (double) dlut_count[se_depth_idx];

                    if (nw_depth == 100.0 || nw_depth > cur_depth + same_range) {
                        for (int c = 0; c < dim_c; c++) {
                            int img_idx = dim3_idx(i, j, c);
                            int nw_img_idx = dim3_idx(y_f, x_f, c);
                            output[nw_img_idx] = img[img_idx];
                        }
                        dlut[nw_depth_idx] = cur_depth;
                        dlut_count[nw_depth_idx] = 1;
                    } else if (std::abs(nw_depth - cur_depth) <= same_range) {
                        for (int c = 0; c < dim_c; c++) {
                            int img_idx = dim3_idx(i, j, c);
                            int nw_img_idx = dim3_idx(y_f, x_f, c);
                            output[nw_img_idx] += img[img_idx];
                        }
                        dlut[nw_depth_idx] += cur_depth;
                        dlut_count[nw_depth_idx]++;
                    }
                    if (ne_depth == 100.0 || ne_depth > cur_depth + same_range) {
                        for (int c = 0; c < dim_c; c++) {
                            int img_idx = dim3_idx(i, j, c);
                            int ne_img_idx = dim3_idx(y_f, x_c, c);
                            output[ne_img_idx] = img[img_idx];
                        }
                        dlut[ne_depth_idx] = cur_depth;
                        dlut_count[ne_depth_idx] = 1;
                    } else if (std::abs(ne_depth - cur_depth) <= same_range) {
                        for (int c = 0; c < dim_c; c++) {
                            int img_idx = dim3_idx(i, j, c);
                            int ne_img_idx = dim3_idx(y_f, x_c, c);
                            output[ne_img_idx] += img[img_idx];
                        }
                        dlut[ne_depth_idx] += cur_depth;
                        dlut_count[ne_depth_idx]++;
                    }
                    if (sw_depth == 100.0 || sw_depth > cur_depth + same_range) {
                        for (int c = 0; c < dim_c; c++) {
                            int img_idx = dim3_idx(i, j, c);
                            int sw_img_idx = dim3_idx(y_c, x_f, c);
                            output[sw_img_idx] = img[img_idx];
                        }
                        dlut[sw_depth_idx] = cur_depth;
                        dlut_count[sw_depth_idx] = 1;
                    } else if (std::abs(sw_depth - cur_depth) <= same_range) {
                        for (int c = 0; c < dim_c; c++) {
                            int img_idx = dim3_idx(i, j, c);
                            int sw_img_idx = dim3_idx(y_c, x_f, c);
                            output[sw_img_idx] += img[img_idx];
                        }
                        dlut[sw_depth_idx] += cur_depth;
                        dlut_count[sw_depth_idx]++;
                    }
                    if (se_depth == 100.0 || se_depth > cur_depth + same_range) {
                        for (int c = 0; c < dim_c; c++) {
                            int img_idx = dim3_idx(i, j, c);
                            int se_img_idx = dim3_idx(y_c, x_c, c);
                            output[se_img_idx] = img[img_idx];
                        }
                        dlut[se_depth_idx] = cur_depth;
                        dlut_count[se_depth_idx] = 1;
                    } else if (std::abs(se_depth - cur_depth) <= same_range) {
                        for (int c = 0; c < dim_c; c++) {
                            int img_idx = dim3_idx(i, j, c);
                            int se_img_idx = dim3_idx(y_c, x_c, c);
                            output[se_img_idx] += img[img_idx];
                        }
                        dlut[se_depth_idx] += cur_depth;
                        dlut_count[se_depth_idx]++;
                    }
                }
            }
        }
        for (int i = 0; i < dim_h; i++) {
            for (int j = 0; j < dim_w; j++) {
                int depth_idx = dim2_idx(i, j);
                if (!dlut_count[depth_idx]) continue;

                for (int c = 0; c < dim_c; c++) {
                    int img_idx = dim3_idx(i, j, c);
                    output[img_idx] = output[img_idx] / (double) dlut_count[depth_idx];
                }
            }
        }

        free(dlut);
        free(dlut_count);
        return;
    }

    void forward_warping_depth(const double *safe_y, const double *safe_x, const double *depth,
                               int dim_h, int dim_w, double same_range, double *output) {
        auto dim2_idx = [&](int y, int x) {
            return y * dim_w + x;
        };

        // double *dlut = (double *)malloc(dim_h * dim_w * sizeof(double));
        // std::fill(dlut, dlut + dim_h * dim_w, 1000.0);
        double *dlut = (double *)malloc(dim_h * dim_w * sizeof(double));
        std::fill(dlut, dlut + dim_h * dim_w, 100.0);
        int *dlut_count = (int *)calloc(dim_h * dim_w, sizeof(int));

        for (int i = 0; i < dim_h; i++) {
            for (int j = 0; j < dim_w; j++) {
                double cur_depth = depth[dim2_idx(i, j)];
                double x = safe_x[dim2_idx(i, j)];
                double y = safe_y[dim2_idx(i, j)];

                int x_f = (int)floor(x);
                int y_f = (int)floor(y);
                int x_c = x_f + 1;
                int y_c = y_f + 1;

                if(x_f >= 0 && x_c < dim_w && y_f >= 0 && y_c < dim_h){
                    int nw_depth_idx = dim2_idx(y_f, x_f);
                    int ne_depth_idx = dim2_idx(y_f, x_c);
                    int sw_depth_idx = dim2_idx(y_c, x_f);
                    int se_depth_idx = dim2_idx(y_c, x_c);

                    double nw_depth = dlut[nw_depth_idx] / (double) dlut_count[nw_depth_idx];
                    double ne_depth = dlut[ne_depth_idx] / (double) dlut_count[ne_depth_idx];
                    double sw_depth = dlut[sw_depth_idx] / (double) dlut_count[sw_depth_idx];
                    double se_depth = dlut[se_depth_idx] / (double) dlut_count[se_depth_idx];

                    if (nw_depth == 100.0 || nw_depth > cur_depth + same_range) {
                        output[nw_depth_idx] = cur_depth;
                        dlut[nw_depth_idx] = cur_depth;
                        dlut_count[nw_depth_idx] = 1;
                    } else if (std::abs(nw_depth - cur_depth) <= same_range) {
                        output[nw_depth_idx] += cur_depth;
                        dlut[nw_depth_idx] += cur_depth;
                        dlut_count[nw_depth_idx]++;
                    }
                    if (ne_depth == 100.0 || ne_depth > cur_depth + same_range) {
                        output[ne_depth_idx] = cur_depth;
                        dlut[ne_depth_idx] = cur_depth;
                        dlut_count[ne_depth_idx] = 1;
                    } else if (std::abs(ne_depth - cur_depth) <= same_range) {
                        output[ne_depth_idx] += cur_depth;
                        dlut[ne_depth_idx] += cur_depth;
                        dlut_count[ne_depth_idx]++;
                    }
                    if (sw_depth == 100.0 || sw_depth > cur_depth + same_range) {
                        output[sw_depth_idx] = cur_depth;
                        dlut[sw_depth_idx] = cur_depth;
                        dlut_count[sw_depth_idx] = 1;
                    } else if (std::abs(sw_depth - cur_depth) <= same_range) {
                        output[sw_depth_idx] += cur_depth;
                        dlut[sw_depth_idx] += cur_depth;
                        dlut_count[sw_depth_idx]++;
                    }
                    if (se_depth == 100.0 || se_depth > cur_depth + same_range) {
                        output[se_depth_idx] = cur_depth;
                        dlut[se_depth_idx] = cur_depth;
                        dlut_count[se_depth_idx] = 1;
                    } else if (std::abs(se_depth - cur_depth) <= same_range) {
                        output[se_depth_idx] += cur_depth;
                        dlut[se_depth_idx] += cur_depth;
                        dlut_count[se_depth_idx]++;
                    }
                }
            }
        }
        for (int i = 0; i < dim_h; i++) {
            for (int j = 0; j < dim_w; j++) {
                int depth_idx = dim2_idx(i, j);
                if (!dlut_count[depth_idx]) continue;

                output[depth_idx] = output[depth_idx] / (double) dlut_count[depth_idx];
            }
        }

        free(dlut);
        free(dlut_count);
        return;
    }

    void forward_warping_flow(const double *flow, const double *safe_y, const double *safe_x, const double *depth,
                              int dim_h, int dim_w, double same_range, double *output) {
        int dim_c = 2;
        auto dim3_idx = [&](int y, int x, int z) {
            return y * dim_c * dim_w + x * dim_c + z;
        };
        auto dim2_idx = [&](int y, int x) {
            return y * dim_w + x;
        };

        double *dlut = (double *)malloc(dim_h * dim_w * sizeof(double));
        std::fill(dlut, dlut + dim_h * dim_w, 100.0);
        int *dlut_count = (int *)calloc(dim_h * dim_w, sizeof(int));

        for (int i = 0; i < dim_h; i++) {
            for (int j = 0; j < dim_w; j++) {
                double cur_depth = depth[dim2_idx(i, j)];
                double x = safe_x[dim2_idx(i, j)];
                double y = safe_y[dim2_idx(i, j)];

                int x_f = (int)floor(x);
                int y_f = (int)floor(y);
                int x_c = x_f + 1;
                int y_c = y_f + 1;

                if(x_f >= 0 && x_c < dim_w && y_f >= 0 && y_c < dim_h){
                    int nw_depth_idx = dim2_idx(y_f, x_f);
                    int ne_depth_idx = dim2_idx(y_f, x_c);
                    int sw_depth_idx = dim2_idx(y_c, x_f);
                    int se_depth_idx = dim2_idx(y_c, x_c);

                    double nw_depth = dlut[nw_depth_idx] / (double) dlut_count[nw_depth_idx];
                    double ne_depth = dlut[ne_depth_idx] / (double) dlut_count[ne_depth_idx];
                    double sw_depth = dlut[sw_depth_idx] / (double) dlut_count[sw_depth_idx];
                    double se_depth = dlut[se_depth_idx] / (double) dlut_count[se_depth_idx];

                    if (nw_depth == 100.0 || nw_depth > cur_depth + same_range) {
                        for (int c = 0; c < dim_c; c++) {
                            int flow_idx = dim3_idx(i, j, c);
                            int nw_flow_idx = dim3_idx(y_f, x_f, c);
                            output[nw_flow_idx] = flow[flow_idx];
                        }
                        dlut[nw_depth_idx] = cur_depth;
                        dlut_count[nw_depth_idx] = 1;
                    } else if (std::abs(nw_depth - cur_depth) <= same_range) {
                        for (int c = 0; c < dim_c; c++) {
                            int flow_idx = dim3_idx(i, j, c);
                            int nw_flow_idx = dim3_idx(y_f, x_f, c);
                            output[nw_flow_idx] += flow[flow_idx];
                        }
                        dlut[nw_depth_idx] += cur_depth;
                        dlut_count[nw_depth_idx]++;
                    }
                    if (ne_depth == 100.0 || ne_depth > cur_depth + same_range) {
                        for (int c = 0; c < dim_c; c++) {
                            int flow_idx = dim3_idx(i, j, c);
                            int ne_flow_idx = dim3_idx(y_f, x_c, c);
                            output[ne_flow_idx] = flow[flow_idx];
                        }
                        dlut[ne_depth_idx] = cur_depth;
                        dlut_count[ne_depth_idx] = 1;
                    } else if (std::abs(ne_depth - cur_depth) <= same_range) {
                        for (int c = 0; c < dim_c; c++) {
                            int flow_idx = dim3_idx(i, j, c);
                            int ne_flow_idx = dim3_idx(y_f, x_c, c);
                            output[ne_flow_idx] += flow[flow_idx];
                        }
                        dlut[ne_depth_idx] += cur_depth;
                        dlut_count[ne_depth_idx]++;
                    }
                    if (sw_depth == 100.0 || sw_depth > cur_depth + same_range) {
                        for (int c = 0; c < dim_c; c++) {
                            int flow_idx = dim3_idx(i, j, c);
                            int sw_flow_idx = dim3_idx(y_c, x_f, c);
                            output[sw_flow_idx] = flow[flow_idx];
                        }
                        dlut[sw_depth_idx] = cur_depth;
                        dlut_count[sw_depth_idx] = 1;
                    } else if (std::abs(sw_depth - cur_depth) <= same_range) {
                        for (int c = 0; c < dim_c; c++) {
                            int flow_idx = dim3_idx(i, j, c);
                            int sw_flow_idx = dim3_idx(y_c, x_f, c);
                            output[sw_flow_idx] += flow[flow_idx];
                        }
                        dlut[sw_depth_idx] += cur_depth;
                        dlut_count[sw_depth_idx]++;
                    }
                    if (se_depth == 100.0 || se_depth > cur_depth + same_range) {
                        for (int c = 0; c < dim_c; c++) {
                            int flow_idx = dim3_idx(i, j, c);
                            int se_flow_idx = dim3_idx(y_c, x_c, c);
                            output[se_flow_idx] = flow[flow_idx];
                        }
                        dlut[se_depth_idx] = cur_depth;
                        dlut_count[se_depth_idx] = 1;
                    } else if (std::abs(se_depth - cur_depth) <= same_range) {
                        for (int c = 0; c < dim_c; c++) {
                            int flow_idx = dim3_idx(i, j, c);
                            int se_flow_idx = dim3_idx(y_c, x_c, c);
                            output[se_flow_idx] += flow[flow_idx];
                        }
                        dlut[se_depth_idx] += cur_depth;
                        dlut_count[se_depth_idx]++;
                    }
                }
            }
        }
        for (int i = 0; i < dim_h; i++) {
            for (int j = 0; j < dim_w; j++) {
                int depth_idx = dim2_idx(i, j);
                if (!dlut_count[depth_idx]) continue;

                for (int c = 0; c < dim_c; c++) {
                    int flow_idx = dim3_idx(i, j, c);
                    output[flow_idx] = output[flow_idx] / (double) dlut_count[depth_idx];
                }
            }
        }

        free(dlut);
        free(dlut_count);
        return;
    }

}
