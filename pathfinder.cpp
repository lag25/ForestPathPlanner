#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>
#include <cmath>
#include <map>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

// Define a tuple for coordinates to be used as a map key
using Point = std::pair<int, int>;

// Custom comparator for the priority queue (min-heap)
struct CompareDist {
    bool operator()(const std::pair<double, Point>& a, const std::pair<double, Point>& b) {
        return a.first > b.first;
    }
};

class OptimalPathing {
public:
    // Constructor
    OptimalPathing() = default;

    // The main function exposed to Python
    std::vector<Point> computeDjikstra(py::array_t<unsigned char> binary_image, Point start_pixel, Point target_pixel) {
        py::buffer_info buf = binary_image.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Input image must be a 2D NumPy array.");
        }

        int rows = buf.shape[0];
        int cols = buf.shape[1];
        auto* ptr = static_cast<unsigned char*>(buf.ptr);

        // --- Dijkstra's Algorithm ---
        std::map<Point, Point> parents;
        std::map<Point, double> costs;
        std::priority_queue<std::pair<double, Point>, std::vector<std::pair<double, Point>>, CompareDist> pq;

        costs[start_pixel] = 0.0;
        pq.push({0.0, start_pixel});

        // Pre-calculate the TCD factor
        double tree_count_density = 115.0 / (rows * cols) * 1000.0;
        double tcd_factor = std::exp(tree_count_density * 100.0);

        while (!pq.empty()) {
            double cost = pq.top().first;
            Point current = pq.top().second;
            pq.pop();
            
            // If we've found a better path already, skip
            if (cost > costs[current] && costs.count(current)) {
                continue;
            }
            
            // Goal check
            if (current == target_pixel) {
                break;
            }

            // --- Explore Neighbors (Graph Creation is implicit) ---
            int i = current.first;
            int j = current.second;

            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    if (dx == 0 && dy == 0) continue;

                    int ni = i + dx;
                    int nj = j + dy;

                    if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                        Point neighbor = {ni, nj};
                        
                        // --- Calculate Edge Weight ---
                        double euclid_dist_to_target = std::sqrt(std::pow(ni - target_pixel.first, 2) + std::pow(nj - target_pixel.second, 2));

                        // Calculate average density
                        double avg_density = 0.0;
                        int density_samples = 0;
                        for (int fact = 1; fact <= 10; ++fact) {
                            int alpha_i = ni - fact * dx;
                            int alpha_j = nj + fact * dy;
                            if (alpha_i >= 0 && alpha_i < rows && alpha_j >= 0 && alpha_j < cols) {
                                avg_density += (255 - ptr[alpha_i * cols + alpha_j]);
                                density_samples++;
                            }

                            int beta_i = ni + fact * dx;
                            int beta_j = nj - fact * dy;
                            if (beta_i >= 0 && beta_i < rows && beta_j >= 0 && beta_j < cols) {
                                avg_density += (255 - ptr[beta_i * cols + beta_j]);
                                density_samples++;
                            }
                        }
                        if (density_samples > 0) {
                            avg_density /= density_samples;
                        }

                        double pixel_intensity_cost = 255 - ptr[ni * cols + nj];
                        double weight = tcd_factor * pixel_intensity_cost + 
                                        std::pow(euclid_dist_to_target, 2) + 
                                        50000.0 * std::log(avg_density + 1.0);
                        
                        double new_cost = cost + weight;

                        if (!costs.count(neighbor) || new_cost < costs[neighbor]) {
                            costs[neighbor] = new_cost;
                            pq.push({new_cost, neighbor});
                            parents[neighbor] = current;
                        }
                    }
                }
            }
        }

        return trace_path(parents, start_pixel, target_pixel);
    }

private:
    // Helper to reconstruct the path
    std::vector<Point> trace_path``(const std::map<Point, Point>& parents, Point start, Point target) {
        std::vector<Point> path;
        Point current = target;
        while (parents.count(current)) {
            path.push_back(current);
            current = parents.at(current);
        }
        path.push_back(start);
        std::reverse(path.begin(), path.end());
        return path;
    }
};

// PYBIND11_MODULE is a macro that creates an entry point for the Python interpreter
PYBIND11_MODULE(pathfinder, m) {
    m.doc() = "High-performance pathfinding module written in C++"; // optional module docstring

    py::class_<OptimalPathing>(m, "OptimalPathing")
        .def(py::init<>()) // Expose the constructor
        .def("compute_path", &OptimalPathing::computeDjikstra, "Computes the optimal path using Dijkstra's algorithm.",
             py::arg("binary_image"), py::arg("start_pixel"), py::arg("target_pixel"));
}
