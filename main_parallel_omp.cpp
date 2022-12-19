#include <cmath>
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <exception>
#include <ctime>
#include <mpi.h>

#include <unistd.h>

MPI_Comm MPI_COMM_CART;

constexpr double EPS = 0.000001;
constexpr double EPS2 = 0.000001;

constexpr double LEFT = -2;
constexpr double RIGHT = 3;
constexpr double BOTTOM = -1;
constexpr double TOP = 4;

constexpr int Ny = 100, Mx = 100;
constexpr double hx = (RIGHT - LEFT) / Mx;
constexpr double hy = (TOP - BOTTOM) / Ny;

using namespace std;

double pow2(double x) {
    return x * x;
}

double pow3(double x) {
    return x * x * x;
}

inline bool eq(double x, double y, double eps = EPS) {
    return fabs(x - y) < eps;
}

class Matrix {
public:
    Matrix(int mx, int ny) {
        size_x = mx + 1;
        actual_size_x = size_x + 2;
        size_y = ny + 1;
        actual_size_y = size_y + 2;

        v = new double *[actual_size_x];
        left.resize(size_y, 0.0);
        left_neighbour.resize(size_y, 0.0);
        right.resize(size_y, 0.0);
        right_neighbour.resize(size_y, 0.0);
        top.resize(size_x, 0.0);
        top_neighbour.resize(size_x, 0.0);
        bottom.resize(size_x, 0.0);
        bottom_neighbour.resize(size_x, 0.0);
        for (int i = 0; i < actual_size_x; ++i) {
            v[i] = new double[actual_size_y];
            for (int j = 0; j < actual_size_y; ++j) {
                v[i][j] = 0.0;
            }
        }
    }

    ~Matrix() {
        for (int i = 0; i < actual_size_x; ++i) {
            delete[] v[i];
        }
        delete[] v;
    }

    int get_size_x() const {
        return size_x;
    }

    int get_size_y() const {
        return size_y;
    }

    double *operator[](int i) {
        return v[i + 1] + 1;
    }

    double operator()(int i, int j) const {
        if (i >= actual_size_x - 1 || j >= actual_size_y - 1) {
            throw "out of range in matrix > actual_size";
        }
        if (i < -1 || j < -1) {
            throw "out of range in matrix < - 1";
        }
        return v[i + 1][j + 1];
    }

    Matrix &operator=(Matrix const &rhs) {
        if (this != &rhs) {
            if (actual_size_x != rhs.actual_size_x || actual_size_y != rhs.actual_size_y) {
                throw "Cant set A = B";
            }
            for (int i = 0; i < actual_size_x; ++i) {
                for (int j = 0; j < actual_size_y; ++j) {
                    v[i][j] = rhs.v[i][j];
                }
            }
        }
        return *this;
    }

    double *get_full() {
        full_matrix.resize(size_x * size_y, 0.0);
        for (int i = 0; i < size_x; ++i) {
            for (int j = 0; j < size_y; ++j) {
                full_matrix[i * size_y + j] = v[i + 1][j + 1];
            }
        }
        return full_matrix.data();
    }

    double *get_full_rec(int mx, int ny) {
        full_matrix.resize((mx + 1) * (ny + 1));
        return full_matrix.data();
    }

    void set_full(int i_start, int j_start, int mx, int ny) {
        for (int i = 0; i < mx + 1; ++i) {
            for (int j = 0; j < ny + 1; ++j) {
                v[i_start + i + 1][j_start + j + 1] = full_matrix[i * (ny + 1) + j];
            }
        }
    }

    double *get_left() {
        for (int i = 0; i < left.size(); ++i) {
            left[i] = v[1][i + 1];
        }
        return left.data();
    }

    double *get_left_rec() {
        return left_neighbour.data();
    }

    void set_left() {
        for (int i = 0; i < left_neighbour.size(); ++i) {
            v[0][i + 1] = left_neighbour[i];
        }
    }

    double *get_right() {
        for (int i = 0; i < right.size(); ++i) {
            right[i] = v[actual_size_x - 2][i + 1];
        }
        return right.data();
    }

    double *get_right_rec() {
        return right_neighbour.data();
    }

    void set_right() {
        for (int i = 0; i < right_neighbour.size(); ++i) {
            v[actual_size_x - 1][i + 1] = right_neighbour[i];
        }
    }

    double *get_bottom() {
        for (int i = 0; i < bottom.size(); ++i) {
            bottom[i] = v[i + 1][1];
        }
        return bottom.data();
    }

    double *get_bottom_rec() {
        return bottom_neighbour.data();
    }

    void set_bottom() {
        for (int i = 0; i < bottom_neighbour.size(); ++i) {
            v[i + 1][0] = bottom_neighbour[i];
        }
    }

    double *get_top() {
        for (int i = 0; i < top.size(); ++i) {
            top[i] = v[i + 1][actual_size_y - 2];
        }
        return top.data();
    }

    double *get_top_rec() {
        return top_neighbour.data();
    }

    void set_top() {
        for (int i = 0; i < top_neighbour.size(); ++i) {
            v[i + 1][actual_size_y - 1] = top_neighbour[i];
        }
    }

private:
    int size_x;
    int actual_size_x;
    int size_y;
    int actual_size_y;

    double **v;
    vector<double> left;
    vector<double> left_neighbour;
    vector<double> right;
    vector<double> right_neighbour;
    vector<double> top;
    vector<double> top_neighbour;
    vector<double> bottom;
    vector<double> bottom_neighbour;
    vector<double> full_matrix;
};

double u(double x, double y) {
    return 2 / (1 + pow2(x) + pow2(y));
}

double du_dx(double x, double y) {
    return -x * pow2(u(x, y));
}

double du_dy(double x, double y) {
    return -x * pow2(u(x, y));
}

double q(double x, double y) {
    return 1;
}

double k(double x, double y) {
    return 1 + pow2(x + y);
}

double F(double x, double y) {
    return - (16.0 * x * x + 16.0 * y * y) / pow3((1 + x * x + y * y)) + 2 * pow2(u(x, y)) + u(x,y);
}

double psi_left(double x, double y) {
    if (eq(x, LEFT)) {
        return k(x, y) * (-du_dx(x, y));
    }
    return 0.0;
}

double psi_right(double x, double y) {
    if (eq(x, RIGHT)) {
        return k(x, y) * du_dx(x, y);
    }
    return 0.0;
}

double psi_bottom(double x, double y) {
    if (eq(y, BOTTOM)) {
        return k(x, y) * (-du_dy(x, y));
    }
    return 0.0;
}

double psi_top(double x, double y) {
    if (eq(y, TOP)) {
        return k(x, y) * du_dy(x, y);
    }
    return 0.0;
}


double rho_x(double x) {
    if (eq(x, LEFT) || eq(x, RIGHT))
        return 0.5;
    else
        return 1.0;
}

double rho_y(double y) {
    if (eq(y, BOTTOM) || eq(y, TOP))
        return 0.5;
    else
        return 1.0;
}

double a(double x, double y) {
    return k(x - 0.5 * hx, y);
}

double b(double x, double y) {
    return k(x, y - 0.5 * hy);
}

double wx_right(const Matrix &omega, int i, int j) {
    return (omega(i + 1, j) - omega(i, j)) / hx;
}

double wx_left(const Matrix &omega, int i, int j) {
    return (omega(i, j) - omega(i - 1, j)) / hx;
}

double wy_right(const Matrix &omega, int i, int j) {
    return (omega(i, j + 1) - omega(i, j)) / hy;
}

double wy_left(const Matrix &omega, int i, int j) {
    return (omega(i, j) - omega(i, j - 1)) / hy;
}

void matrix_diff(Matrix &res, Matrix &u, Matrix &v, int i_start, int j_start, bool tt=true, double tau = 1.0) {
    #pragma omp parallel for
    for (int i = 0; i < res.get_size_x(); ++i) {
        #pragma omp parallel for
        for (int j = 0; j < res.get_size_y(); ++j) {
            if (i + i_start == 0 || i + i_start == Mx || j + j_start == 0 || j + j_start == Ny) {
                if (tt)
                    res[i][j] = 0;
                else
                    res[i][j] = u[i][j];
            } else
                res[i][j] = u(i, j) - tau * v(i, j);
        }
    }
}

double scalar(Matrix &u, Matrix &v, int i_start, int j_start) {
    double ans = 0.0;
    #pragma omp parallel for reduction(+:ans)
    for (int i = 0; i < u.get_size_x(); ++i) {
        #pragma omp parallel for reduction(+:ans)
        for (int j = 0; j < u.get_size_y(); ++j) {
            if (i + i_start == 0 || i + i_start == Mx || j + j_start == 0 || j + j_start == Ny) {

            } else {
                ans += rho_x(LEFT + (i + i_start) * hx) * rho_y(BOTTOM + (j + j_start) * hy) * u(i, j) * v(i, j);
            }
        }
    }
    return ans * hx * hy;
}

double norm(Matrix &u, int i_start, int j_start) {
    return sqrt(scalar(u, u, i_start, j_start));
}

void getB(Matrix &B, int i_start, int j_start) {
    double psi_00 = (2.0 / hx + 2.0 / hy) * (hx * psi_bottom(LEFT, BOTTOM) + hy * psi_left(LEFT, BOTTOM)) / (hx + hy);
    double psi_0N = (2.0 / hx + 2.0 / hy) * (hx * psi_top(LEFT, TOP) + hy * psi_left(LEFT, TOP)) / (hx + hy);
    double psi_M0 =
            (2.0 / hx + 2.0 / hy) * (hx * psi_bottom(RIGHT, BOTTOM) + hy * psi_right(RIGHT, BOTTOM)) / (hx + hy);
    double psi_MN = (2.0 / hx + 2.0 / hy) * (hx * psi_top(RIGHT, TOP) + hy * psi_right(RIGHT, TOP)) / (hx + hy);
    #pragma omp parallel for
    for (int i = 0; i < B.get_size_x(); ++i) {
        #pragma omp parallel for
        for (int j = 0; j < B.get_size_y(); ++j) {
            B[i][j] = F(LEFT + (i + i_start) * hx, BOTTOM + (j + j_start) * hy);
        }
    }
    // нижняя граница
    if (j_start == 0) {
        #pragma omp parallel for
        for (int i = 0; i < B.get_size_x(); ++i) {
            B[i][0] += 2.0 / hy * psi_bottom(LEFT + (i + i_start) * hx, BOTTOM);
        }
    }
    // верхняя граница
    if (j_start + B.get_size_y() - 1 == Ny) {
        #pragma omp parallel for
        for (int i = 0; i < B.get_size_x(); ++i) {
            B[i][B.get_size_y() - 1] += 2.0 / hy * psi_top(LEFT + (i + i_start) * hx, TOP);
        }
    }

    // Левая граница
    if (i_start == 0) {
        #pragma omp parallel for
        for (int j = 0; j < B.get_size_y(); ++j) {
            B[0][j] += 2.0 / hx * psi_left(LEFT, BOTTOM + (j + j_start) * hy);
        }
    }

    // правая граница
    if (i_start + B.get_size_x() - 1 == Mx) {
        #pragma omp parallel for
        for (int j = 0; j < B.get_size_y(); ++j) {
            B[B.get_size_x() - 1][j] += 2.0 / hx * psi_right(RIGHT, BOTTOM + (j + j_start) * hy);
        }
    }

    // левый нижний угол
    if (i_start == 0 && j_start == 0)
        B[0][0] = F(LEFT, BOTTOM) + psi_00;
    // левый верхний угол
    if (i_start == 0 && j_start + B.get_size_y() - 1 == Ny)
        B[0][B.get_size_y()] = F(LEFT, TOP) + psi_0N;
    // правый нижний угол
    if (i_start + B.get_size_x() - 1 == Mx && j_start == 0)
        B[B.get_size_x()][0] = F(RIGHT, BOTTOM) + psi_M0;
    // правый верхний угол
    if (i_start + B.get_size_x() - 1 == Mx && j_start + B.get_size_y() - 1 == Ny)
        B[B.get_size_x()][B.get_size_y()] = F(RIGHT, TOP) + psi_MN;
}

void Aw(Matrix &Aomega, Matrix &omega, int i_start, int j_start) {
    #pragma omp parallel for
    for (int i = 0; i < omega.get_size_x(); ++i) {
        #pragma omp parallel for
        for (int j = 0; j < omega.get_size_y(); ++j) {
            double omega_x_left = 0.0;
            double omega_x_right = 0.0;
            double omega_y_left = 0.0;
            double omega_y_right = 0.0;
            double x = LEFT + (i + i_start) * hx;
            double y = BOTTOM + (j + j_start) * hy;
            if (i + i_start != Mx) {
                omega_x_right = a(x + hx, y) * wx_right(omega, i, j);
            }
            if (i + i_start != 0) {
                omega_x_left = -a(x, y) * wx_left(omega, i, j);
            }
            if (j + j_start != Ny) {
                omega_y_right = b(x, y + hy) * wy_right(omega, i, j);
            }
            if (j + j_start != 0) {
                omega_y_left = -b(x, y) * wy_left(omega, i, j);
            }

            double omega_x = omega_x_left + omega_x_right;
            double omega_x_x = omega_x / hx;
            double omega_y = omega_y_left + omega_y_right;
            double omega_y_y = omega_y / hy;

            // (bw_y-)_i_j
            double b_omega_y = b(x, y) * wy_left(omega, i, j);
            // для (bw_y-)_i_1
            double b_omega_y_right = b(x, y) * wy_right(omega, i, j);
            // (aw_x-)_i_j
            double a_omega_x = a(x, y) * wx_left(omega, i, j);
            // для (aw_x-)_1_j
            double a_omega_x_right = a(x, y) * wx_right(omega, i, j);

            if (j + j_start == Ny && i + i_start != 0 && i + i_start != Mx) {
                // верхняя граница
                Aomega[i][j] = 2.0 / hy * b_omega_y + q(x, y) * omega(i, j) - omega_x_x;
            } else if (j + j_start == 0 && i + i_start != 0 && i + i_start != Mx) {
                // нижняя граница
                Aomega[i][j] = -2.0 / hy * b_omega_y_right + q(x, y) * omega(i, j) - omega_x_x;
            } else if (i + i_start == Mx && j + j_start != 0 && j + j_start != Ny) {
                // правая граница
                Aomega[i][j] = 2.0 / hx * a_omega_x + q(x, y) * omega(i, j) - omega_y_y;
            } else if (i + i_start == 0 && j + j_start != 0 && j + j_start != Ny) {
                // левая граница
                Aomega[i][j] = -2.0 / hx * a_omega_x_right + q(x, y) * omega(i, j) - omega_y_y;
            } else if (i + i_start == 0 && j + j_start == 0) {
                // левый нижний угол
                Aomega[i][j] = -2.0 / hx * a_omega_x_right - 2.0 / hy * b_omega_y_right +
                               q(x, y) * omega(i, j);
            } else if (i + i_start == Mx && j + j_start == 0) {
                // правый нижний угол
                Aomega[i][j] = 2.0 / hx * a_omega_x - 2.0 / hy * b_omega_y_right +
                               q(x, y) * omega(i, j);
            } else if (i + i_start == Mx && j + j_start == Ny) {
                // правый верхний угол
                Aomega[i][j] =
                        2.0 / hx * a_omega_x + 2.0 / hy * b_omega_y + q(x, y) * omega(i, j);
            } else if (i + i_start == 0 && j + j_start == Ny) {
                // левый верхний угол
                Aomega[i][j] = -2.0 / hx * a_omega_x_right + 2.0 / hy * b_omega_y +
                               q(x, y) * omega(i, j);
            } else {
                Aomega[i][j] = -omega_x_x - omega_y_y + q(x, y) * omega(i, j);
            }
        }
    }
}

void print_matrix(Matrix& m, string name, bool full=false) {
    cout << name << endl;
    int st = (full) ? -1 : 0;
    int end = (full) ? 1 : 0;
    for (int i = 0 + st; i < m.get_size_x() + end; ++i) {
        for (int j = 0 + st; j < m.get_size_y() + end; ++j) {
            cout << fixed << m(i, j) << " ";
        }
        cout << endl;
    }
    cout << "==================" << endl;
}

void get_idx(int *grid_size, int *coords, int *i_start, int *j_start, int *mx, int *ny) {
    *mx = (Mx + 1) / grid_size[0];
    bool i_add_condition = ((Mx + 1) % grid_size[0] > coords[0]);
    int i_shift = min((Mx + 1) % grid_size[0], coords[0]);
    *mx += (i_add_condition) ? 1 : 0;
    *i_start = *mx * coords[0] + i_shift;
    *ny = (Ny + 1) / grid_size[1];
    bool j_add_condition = ((Ny + 1) % grid_size[1] > coords[1]);
    int j_shift = min((Ny + 1) % grid_size[1], coords[1]);
    *j_start = *ny * coords[1] + j_shift;
    *ny += (j_add_condition) ? 1 : 0;
    (*mx)--;
    (*ny)--;
}

bool get_neighbour(int* coords, int* grid_size, int* neighbour_rank, char side) {
    int neighbour_coords[2] = {coords[0], coords[1]};
    if (side == 'l') {
        neighbour_coords[0]--;
    } else if (side == 'r') {
        neighbour_coords[0]++;
    } else if (side == 't') {
        neighbour_coords[1]++;
    } else if (side == 'b') {
        neighbour_coords[1]--;
    } else {
        cout << "PANIC 4" << endl;
        throw "unknown";
    }
    if (neighbour_coords[0] < 0 || neighbour_coords[0] >= grid_size[0])
        return false;
    if (neighbour_coords[1] < 0 || neighbour_coords[1] >= grid_size[1])
        return false;
    MPI_Cart_rank(MPI_COMM_CART, neighbour_coords, neighbour_rank);
    return true;
}

int main(int argc, char** argv) {
    int rank;
    int commSize;
    int neighbour_rank;
    int grid_size[2] = {0};

    MPI_Status status;
    MPI_Request request;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Dims_create(commSize, 2, grid_size);

    int periods[2] = {0};
    int coords[2];
    int neighbour_coords[2];

    MPI_Cart_create(MPI_COMM_WORLD, 2, grid_size, periods, true, & MPI_COMM_CART);
    int grid_rank;
    MPI_Comm_rank(MPI_COMM_CART, &grid_rank);
    MPI_Cart_coords(MPI_COMM_CART, grid_rank, 2, coords);
    int tag = 777;
    int tag2 = 666;

    double start_time = MPI_Wtime();


    int mx, ny, i_start, j_start;
    get_idx(grid_size, coords, &i_start, &j_start, &mx, &ny);

    Matrix omega(mx, ny);
    Matrix omega_new(mx, ny);
    Matrix B(mx, ny);
    Matrix r(mx, ny);
    Matrix Aomega(mx, ny);
    Matrix Ar(mx, ny);
    Matrix omega_error(mx, ny);
    double tau;
    getB(B, i_start, j_start);
    double diff = 5;
    double numerator_global = 0;
    double denumerator_global = 0;


    int iterations = 0;


    while (diff > EPS2) {
        if (get_neighbour(coords, grid_size, &neighbour_rank, 'l')) {
            MPI_Isend(omega.get_left(), omega.get_size_y(), MPI_DOUBLE, neighbour_rank, tag, MPI_COMM_CART, &request);
        }
        if (get_neighbour(coords, grid_size, &neighbour_rank, 'r')) {
            MPI_Isend(omega.get_right(), omega.get_size_y(), MPI_DOUBLE, neighbour_rank, tag, MPI_COMM_CART, &request);
        }
        if (get_neighbour(coords, grid_size, &neighbour_rank, 'b')) {
            MPI_Isend(omega.get_bottom(), omega.get_size_x(), MPI_DOUBLE, neighbour_rank, tag, MPI_COMM_CART, &request);
        }
        if (get_neighbour(coords, grid_size, &neighbour_rank, 't')) {
            MPI_Isend(omega.get_top(), omega.get_size_x(), MPI_DOUBLE, neighbour_rank, tag, MPI_COMM_CART, &request);
        }
        if (get_neighbour(coords, grid_size, &neighbour_rank, 'l')) {
            MPI_Recv(omega.get_left_rec(), omega.get_size_y(), MPI_DOUBLE, neighbour_rank, tag, MPI_COMM_CART, &status);
            omega.set_left();
        }
        if (get_neighbour(coords, grid_size, &neighbour_rank, 'r')) {
            MPI_Recv(omega.get_right_rec(), omega.get_size_y(), MPI_DOUBLE, neighbour_rank, tag, MPI_COMM_CART, &status);
            omega.set_right();
        }
        if (get_neighbour(coords, grid_size, &neighbour_rank, 'b')) {
            MPI_Recv(omega.get_bottom_rec(), omega.get_size_x(), MPI_DOUBLE, neighbour_rank, tag, MPI_COMM_CART, &status);
            omega.set_bottom();
        }
        if (get_neighbour(coords, grid_size, &neighbour_rank, 't')) {
            MPI_Recv(omega.get_top_rec(), omega.get_size_x(), MPI_DOUBLE, neighbour_rank, tag, MPI_COMM_CART, &status);
            omega.set_top();
        }


        Aw(Aomega, omega, i_start, j_start);
        matrix_diff(r, Aomega, B, i_start, j_start);

        if (get_neighbour(coords, grid_size, &neighbour_rank, 'l')) {
            MPI_Isend(r.get_left(), r.get_size_y(), MPI_DOUBLE, neighbour_rank, tag2, MPI_COMM_CART, &request);
        }
        if (get_neighbour(coords, grid_size, &neighbour_rank, 'r')) {
            MPI_Isend(r.get_right(), r.get_size_y(), MPI_DOUBLE, neighbour_rank, tag2, MPI_COMM_CART, &request);
        }
        if (get_neighbour(coords, grid_size, &neighbour_rank, 'b')) {
            MPI_Isend(r.get_bottom(), r.get_size_x(), MPI_DOUBLE, neighbour_rank, tag2, MPI_COMM_CART, &request);
        }
        if (get_neighbour(coords, grid_size, &neighbour_rank, 't')) {
            MPI_Isend(r.get_top(), r.get_size_x(), MPI_DOUBLE, neighbour_rank, tag2, MPI_COMM_CART, &request);
        }
        if (get_neighbour(coords, grid_size, &neighbour_rank, 'l')) {
            MPI_Recv(r.get_left_rec(), r.get_size_y(), MPI_DOUBLE, neighbour_rank, tag2, MPI_COMM_CART, &status);
            r.set_left();
        }
        if (get_neighbour(coords, grid_size, &neighbour_rank, 'r')) {
            MPI_Recv(r.get_right_rec(), r.get_size_y(), MPI_DOUBLE, neighbour_rank, tag2, MPI_COMM_CART, &status);
            r.set_right();
        }
        if (get_neighbour(coords, grid_size, &neighbour_rank, 'b')) {
            MPI_Recv(r.get_bottom_rec(), r.get_size_x(), MPI_DOUBLE, neighbour_rank, tag2, MPI_COMM_CART, &status);
            r.set_bottom();
        }
        if (get_neighbour(coords, grid_size, &neighbour_rank, 't')) {
            MPI_Recv(r.get_top_rec(), r.get_size_x(), MPI_DOUBLE, neighbour_rank, tag2, MPI_COMM_CART, &status);
            r.set_top();
        }

        Aw(Ar, r, i_start, j_start);

        double numerator = scalar(Ar, r, i_start, j_start);
        double denumerator = pow2(norm(Ar, i_start, j_start));
        MPI_Allreduce(&numerator, &numerator_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_CART);
        MPI_Allreduce(&denumerator, &denumerator_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_CART);
        tau = numerator_global / denumerator_global;

        matrix_diff(omega_new, omega, r, i_start, j_start, false, tau);
        matrix_diff(omega_error, omega_new, omega, i_start, j_start);
        omega = omega_new;
        double diff_local = norm(omega_error, i_start, j_start);
        MPI_Allreduce(&diff_local, &diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_CART);
        iterations++;
    }


    double end_time = MPI_Wtime();

    if (grid_rank == 0) {
        Matrix full_omega(Mx, Ny);
        for (int i = 0; i < omega.get_size_x(); ++i) {
            for (int j = 0; j < omega.get_size_y(); ++j) {
                full_omega[i + i_start][j + j_start] = omega[i][j];
            }
        }
        for (int proc = 1; proc < commSize; ++proc) {
            int coords2[2];
            MPI_Cart_coords(MPI_COMM_CART, proc, 2, coords2);
            int mx_temp, ny_temp, i_start_temp, j_start_temp;
            get_idx(grid_size, coords2, &i_start_temp, &j_start_temp, &mx_temp, &ny_temp);

            MPI_Recv(full_omega.get_full_rec(mx_temp, ny_temp), (mx_temp + 1) * (ny_temp + 1), MPI_DOUBLE, proc, 999, MPI_COMM_CART, &status);
            full_omega.set_full(i_start_temp, j_start_temp, mx_temp, ny_temp);

        }
        FILE* f;
        f = fopen("matrix_parallel.txt", "w");
        fprintf(f, "%d\n", Mx);
        fprintf(f, "%d\n", Ny);
        diff = 0;
        for (int i = 0; i < full_omega.get_size_x(); i++) {
            for (int j = 0; j < full_omega.get_size_y(); j++) {
                diff += rho_x(LEFT + i * hx) * rho_y(BOTTOM + j * hy) * pow2(full_omega(i,j) - u(LEFT + i * hx, BOTTOM + j * hy));
                fprintf(f, "%lf %lf %lf\n", LEFT + i * hx, BOTTOM + j * hy, full_omega(i,j));
            }
        }
        diff *= hx * hy;
        fclose(f);
    } else {

        MPI_Isend(omega.get_full(), (mx + 1) * (ny + 1), MPI_DOUBLE, 0, 999, MPI_COMM_CART, &request);
    }

    if (grid_rank == 0) {
        cout << "iterations: " << iterations << endl;
        cout << "wtime: " << end_time - start_time << endl;
        cout << "error: " << diff << endl;
    }
    MPI_Finalize();
    return 0;
}