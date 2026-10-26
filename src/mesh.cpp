#include <CGAL/Bbox_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Image_3.h>
#include <CGAL/Labeled_mesh_domain_3.h>
#include <CGAL/Mesh_3/Image_plus_weights_to_labeled_function_wrapper.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/make_mesh_3.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace parameters = CGAL::parameters;

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;
using MeshDomain = CGAL::Labeled_mesh_domain_3<Kernel>;
using ConcurrencyTag = CGAL::Parallel_tag;
using Triangulation = CGAL::Mesh_triangulation_3<
    MeshDomain,
    CGAL::Default,
    ConcurrencyTag>::type;
using C3t3 = CGAL::Mesh_complex_3_in_triangulation_3<Triangulation>;
using MeshCriteria = CGAL::Mesh_criteria_3<Triangulation>;

using LabelImage = CGAL::Image_3;
using WeightedLabelFunction =
    CGAL::Mesh_3::Image_plus_weights_to_labeled_function_wrapper<
        std::int32_t, int, unsigned char, int>;

/** Build a padded CGAL label image in the input volume's voxel coordinates. */
LabelImage make_label_image(
    const std::int32_t* labels,
    const std::array<std::size_t, 3>& shape,
    const double* affine) {
    const double spacing_x = std::hypot(affine[2], affine[6], affine[10]);
    const double spacing_y = std::hypot(affine[1], affine[5], affine[9]);
    const double spacing_z = std::hypot(affine[0], affine[4], affine[8]);
    _image* raw_image = _createImage(
        shape[2] + 2,
        shape[1] + 2,
        shape[0] + 2,
        1,
        spacing_x,
        spacing_y,
        spacing_z,
        sizeof(std::int32_t),
        WK_FIXED,
        SGN_SIGNED);
    if (raw_image == nullptr || raw_image->data == nullptr) {
        if (raw_image != nullptr) {
            _freeImage(raw_image);
        }
        throw std::runtime_error("CGAL could not allocate a label image.");
    }
    LabelImage image(raw_image);
    auto* image_data = static_cast<std::int32_t*>(image.data());
    const std::size_t padded_x = shape[2] + 2;
    const std::size_t padded_y = shape[1] + 2;
    std::fill(image_data, image_data + padded_x * padded_y * (shape[0] + 2), 0);
    for (std::size_t i = 0; i < shape[0]; ++i) {
        for (std::size_t j = 0; j < shape[1]; ++j) {
            for (std::size_t k = 0; k < shape[2]; ++k) {
                const std::size_t source = (i * shape[1] + j) * shape[2] + k;
                const std::size_t target =
                    ((i + 1) * padded_y + (j + 1)) * padded_x + (k + 1);
                image_data[target] = labels[source];
            }
        }
    }
    image.image()->tx = 0.0f;
    image.image()->ty = 0.0f;
    image.image()->tz = 0.0f;
    return image;
}

LabelImage make_label_weights(
    const LabelImage& labels,
    double sigma) {
    using Weight = unsigned char;
    const std::size_t size = labels.size();
    const std::size_t dim_x = labels.xdim();
    const std::size_t dim_y = labels.ydim();
    const std::size_t dim_z = labels.zdim();
    const std::size_t dim_xy = dim_x * dim_y;
    const auto* label_data = static_cast<const std::int32_t*>(labels.data());
    const std::set<std::int32_t> unique_labels(label_data, label_data + size);

    _image* raw_weights = _createImage(
        dim_x,
        dim_y,
        dim_z,
        1,
        labels.vx(),
        labels.vy(),
        labels.vz(),
        sizeof(Weight),
        WK_FIXED,
        SGN_UNSIGNED);
    if (raw_weights == nullptr || raw_weights->data == nullptr) {
        if (raw_weights != nullptr) {
            _freeImage(raw_weights);
        }
        throw std::runtime_error("CGAL could not allocate a label-weights image.");
    }
    LabelImage weights(raw_weights);
    weights.image()->tx = labels.image()->tx;
    weights.image()->ty = labels.image()->ty;
    weights.image()->tz = labels.image()->tz;
    auto* output = static_cast<Weight*>(weights.data());
    std::fill(output, output + size, Weight{0});

    const std::array<double, 3> spacing = {
        labels.vz(), labels.vy(), labels.vx()};
    const std::array<std::size_t, 3> dimensions = {dim_z, dim_y, dim_x};
    const std::array<std::size_t, 3> strides = {dim_xy, dim_x, 1};
    std::array<std::vector<double>, 3> kernels;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const double voxel_sigma = sigma / spacing[axis];
        const std::size_t radius = static_cast<std::size_t>((std::min)(
            std::ceil(4.0 * voxel_sigma),
            static_cast<double>(dimensions[axis] - 1)));
        kernels[axis].resize(radius * 2 + 1);
        double sum = 0.0;
        for (std::size_t offset = 0; offset < kernels[axis].size(); ++offset) {
            const double delta = static_cast<double>(offset) - radius;
            const double value = delta == 0.0
                ? 1.0
                : std::exp(-0.5 * delta * delta / (voxel_sigma * voxel_sigma));
            kernels[axis][offset] = value;
            sum += value;
        }
        for (double& value : kernels[axis]) {
            value /= sum;
        }
    }
    std::vector<double> first(size);
    std::vector<double> second(size);
    for (const std::int32_t label : unique_labels) {
        for (std::size_t i = 0; i < size; ++i) {
            first[i] = label_data[i] == label ? 1.0 : 0.0;
        }
        const std::vector<double>* source = &first;
        std::vector<double>* target = &second;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const std::size_t stride = strides[axis];
            const std::size_t dimension = dimensions[axis];
            const std::size_t radius = kernels[axis].size() / 2;
            for (std::size_t index = 0; index < size; ++index) {
                const std::size_t coordinate = (index / stride) % dimension;
                double value = 0.0;
                for (std::size_t kernel_index = 0;
                     kernel_index < kernels[axis].size();
                     ++kernel_index) {
                    const std::ptrdiff_t neighbor =
                        static_cast<std::ptrdiff_t>(coordinate) +
                        static_cast<std::ptrdiff_t>(kernel_index) -
                        static_cast<std::ptrdiff_t>(radius);
                    if (neighbor >= 0 && neighbor <
                            static_cast<std::ptrdiff_t>(dimension)) {
                        const auto source_index = static_cast<std::size_t>(
                            static_cast<std::ptrdiff_t>(index) +
                            (neighbor - static_cast<std::ptrdiff_t>(coordinate)) *
                                static_cast<std::ptrdiff_t>(stride));
                        value += (*source)[source_index] * kernels[axis][kernel_index];
                    }
                }
                (*target)[index] = value;
            }
            const std::vector<double>* previous_source = source;
            source = target;
            target = previous_source == &first ? &first : &second;
        }
        for (std::size_t i = 0; i < size; ++i) {
            const auto value = static_cast<Weight>(std::clamp(
                std::lround((*source)[i] * 255.0), 0L, 255L));
            output[i] = (std::max)(output[i], value);
        }
    }
    return weights;
}

struct WeightedLabelView {
    LabelImage labels;
    LabelImage weights;
    std::array<std::size_t, 3> shape;
    const double* world_to_voxel;

    int operator()(const Point& point) const {
        const double x = CGAL::to_double(point.x());
        const double y = CGAL::to_double(point.y());
        const double z = CGAL::to_double(point.z());
        const std::array<double, 3> voxel = {
            world_to_voxel[0] * x + world_to_voxel[1] * y +
                world_to_voxel[2] * z + world_to_voxel[3],
            world_to_voxel[4] * x + world_to_voxel[5] * y +
                world_to_voxel[6] * z + world_to_voxel[7],
            world_to_voxel[8] * x + world_to_voxel[9] * y +
                world_to_voxel[10] * z + world_to_voxel[11],
        };
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (voxel[axis] < -0.5 ||
                voxel[axis] >= static_cast<double>(shape[axis]) - 0.5) {
                return 0;
            }
        }
        const WeightedLabelFunction labeling(labels, weights);
        return labeling(
            Point((voxel[2] + 1.0) * labels.vx(),
                  (voxel[1] + 1.0) * labels.vy(),
                  (voxel[0] + 1.0) * labels.vz()));
    }
};

/**
 * Generate a tetrahedral mesh for a labelled volume.
 *
 * `labels` is a contiguous 3-D int32 volume, `world_to_voxel` and `affine` are
 * its inverse and forward affines, and `bounds` contains the world-space box.
 * `smoothing_sigma` controls CGAL's weighted-label boundary smoothing.
 */
py::tuple create(
    py::array_t<std::int32_t, py::array::c_style> labels,
    py::array_t<double, py::array::c_style> world_to_voxel,
    py::array_t<double, py::array::c_style> affine,
    py::array_t<double, py::array::c_style> bounds,
    double tetrahedra_size,
    double distance,
    double smoothing_sigma) {
    const py::buffer_info buffer = labels.request();
    const py::buffer_info affine_buffer = world_to_voxel.request();
    const py::buffer_info forward_affine_buffer = affine.request();
    const py::buffer_info bounds_buffer = bounds.request();
    const auto* inverse_affine =
        static_cast<const double*>(affine_buffer.ptr);
    const auto* domain_bounds = static_cast<const double*>(bounds_buffer.ptr);
    const std::array<std::size_t, 3> shape = {
        static_cast<std::size_t>(buffer.shape[0]),
        static_cast<std::size_t>(buffer.shape[1]),
        static_cast<std::size_t>(buffer.shape[2]),
    };
    const CGAL::Bbox_3 domain_bbox(
        domain_bounds[0],
        domain_bounds[1],
        domain_bounds[2],
        domain_bounds[3],
        domain_bounds[4],
        domain_bounds[5]);

    LabelImage label_image;
    LabelImage weights;
    {
        py::gil_scoped_release release;
        label_image = make_label_image(
            static_cast<const std::int32_t*>(buffer.ptr),
            shape,
            static_cast<const double*>(forward_affine_buffer.ptr));
        weights = make_label_weights(label_image, smoothing_sigma);
    }
    WeightedLabelView view{
        label_image, weights, shape, inverse_affine};
    MeshDomain domain(view, domain_bbox);
    MeshCriteria criteria(
        parameters::cell_size(tetrahedra_size).facet_distance(distance));
    C3t3 mesh;
    {
        // Meshing can be expensive and does not access Python objects, so let
        // other Python threads run while CGAL constructs the mesh.
        py::gil_scoped_release release;
        mesh = CGAL::make_mesh_3<C3t3>(domain, criteria);
    }

    using VertexHandle = Triangulation::Vertex_handle;
    const auto begin = mesh.cells_in_complex_begin();
    const auto end = mesh.cells_in_complex_end();
    if (begin == end) {
        throw std::runtime_error("CGAL generated an empty tetrahedral mesh.");
    }
    std::map<VertexHandle, std::size_t> vertex_indices;
    std::size_t tetrahedron_count = 0;
    // Assign each distinct mesh vertex one compact array index while counting
    // cells. The map also lets the output pass reuse those indices per cell.
    for (auto cell = begin; cell != end; ++cell) {
        ++tetrahedron_count;
        for (int local_index = 0; local_index < 4; ++local_index) {
            vertex_indices.emplace(cell->vertex(local_index), vertex_indices.size());
        }
    }
    if (vertex_indices.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("The mesh has too many vertices for uint32 indices.");
    }
    if (vertex_indices.size() >
            static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max()) ||
        tetrahedron_count >
            static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max())) {
        throw std::overflow_error("The mesh is too large for a NumPy array.");
    }

    py::array_t<double> vertices({
        static_cast<py::ssize_t>(vertex_indices.size()), py::ssize_t{3}});
    py::array_t<std::uint32_t> tetrahedra({
        static_cast<py::ssize_t>(tetrahedron_count), py::ssize_t{4}});
    py::array_t<std::int32_t> cell_labels({
        static_cast<py::ssize_t>(tetrahedron_count)});
    auto vertex_output = vertices.mutable_unchecked<2>();
    auto tetrahedron_output = tetrahedra.mutable_unchecked<2>();
    auto label_output = cell_labels.mutable_unchecked<1>();

    // Emit point coordinates in the exact order established by vertex_indices.
    for (const auto& [vertex, index] : vertex_indices) {
        const Point& point = vertex->point().point();
        vertex_output(index, 0) = CGAL::to_double(point.x());
        vertex_output(index, 1) = CGAL::to_double(point.y());
        vertex_output(index, 2) = CGAL::to_double(point.z());
    }
    std::size_t tetrahedron_index = 0;
    for (auto cell = begin; cell != end; ++cell, ++tetrahedron_index) {
        for (int local_index = 0; local_index < 4; ++local_index) {
            tetrahedron_output(tetrahedron_index, local_index) =
                static_cast<std::uint32_t>(vertex_indices.at(cell->vertex(local_index)));
        }
        label_output(tetrahedron_index) =
            static_cast<std::int32_t>(cell->subdomain_index());
    }
    return py::make_tuple(
        std::move(vertices), std::move(tetrahedra), std::move(cell_labels));
}

PYBIND11_MODULE(_mesh, module) {
    module.doc() =
        "Private CGAL Mesh_3 bindings for labelled MRI volumes. "
        "Used by the public ``mri2tet.tetrahedralize`` interface.";
    module.def(
        "create",
        &create,
        py::arg("labels").noconvert(),
        py::arg("world_to_voxel").noconvert(),
        py::arg("affine").noconvert(),
        py::arg("bounds").noconvert(),
        py::arg("tetrahedra_size"),
        py::arg("distance"),
        py::arg("smoothing_sigma"),
        "Generate a tetrahedral mesh and retain integer subdomain labels. "
        "This internal function expects validated, contiguous NumPy arrays.");
}
