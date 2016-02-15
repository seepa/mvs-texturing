/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <set>
#include <map>

#include <util/file_system.h>
#include <mve/image_tools.h>
#include <mve/image_io.h>

#include "texture_atlas.h"

TextureAtlas::TextureAtlas(unsigned int size) :
    size(size), padding(size >> 7), finalized(false) {

    bin = RectangularBin::create(size, size);
    image = mve::ByteImage::create(size, size, 3);
    validity_mask = mve::ByteImage::create(size, size, 1);
}


typedef std::vector<std::pair<int, int> > PixelVector;

bool
TextureAtlas::insert(TexturePatch::ConstPtr texture_patch, float mean, float max) {
    if (finalized) {
        throw util::Exception("No insertion possible, TextureAtlas already finalized");
    }

    assert(bin != nullptr);
    assert(validity_mask != nullptr);

    int const width = texture_patch->get_width();
    int const height = texture_patch->get_height();
    Rect<int> rect(0, 0, width + 2 * padding, height + 2 * padding);
    if (!bin->insert(&rect)) return false;

    float max_2 = max * max;

    /* Update texture atlas and its validity mask. */
    mve::FloatImage::ConstPtr opatch_image = texture_patch->get_image();
    mve::ByteImage::Ptr patch_image = mve::ByteImage::create(width, height, 3);
    mve::ByteImage::ConstPtr patch_validity_mask = texture_patch->get_validity_mask();
    for (int i = 0; i < patch_image->get_value_amount(); ++i) {
        if (patch_validity_mask->at(i / 3) == 0) continue;

        // TODO: Investigate effect of clamping negative values
        float v = std::max(opatch_image->at(i), 0.0f);

        /* Apply tone mapping as proposed in
         * Reinhard, Erik, et al. "Photographic tone reproduction for digital images."
         * Transactions on Graphics (TOG). Vol. 21. No. 3. ACM, 2002.
         */
        v = (0.18f / mean) * v;
        v = (v * (1.0f + v / max_2)) / (1.0f + v);
        patch_image->at(i) = std::max(0.0f, std::min(v * 255.0f, 255.0f));
    }
    mve::image::gamma_correct(patch_image, 1.0f / 2.2f);

    copy_into<uint8_t>(patch_image, rect.min_x, rect.min_y, image, padding);
    copy_into<uint8_t>(patch_validity_mask, rect.min_x, rect.min_y, validity_mask, padding);

    TexturePatch::Faces const & patch_faces = texture_patch->get_faces();
    TexturePatch::Texcoords const & patch_texcoords = texture_patch->get_texcoords();

    /* Calculate the offset of the texture patches' relative texture coordinates */
    math::Vec2f offset = math::Vec2f(rect.min_x + padding, rect.min_y + padding);

    faces.insert(faces.end(), patch_faces.begin(), patch_faces.end());

    /* Calculate the final textcoords of the faces. */
    for (std::size_t i = 0; i < patch_faces.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            math::Vec2f rel_texcoord(patch_texcoords[i * 3 + j]);
            math::Vec2f texcoord = rel_texcoord + offset;

            texcoord[0] = texcoord[0] / this->size;
            texcoord[1] = texcoord[1] / this->size;
            texcoords.push_back(texcoord);
        }
    }
    return true;
}

inline bool
has_valid_neighbor(mve::ByteImage::Ptr validity_mask, int x, int y) {
    int width = validity_mask->width();
    int height = validity_mask->height();

    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int nx = x + i;
            int ny = y + j;

            if (nx < 0 || width <= nx) continue;
            if (ny < 0 || height <= ny) continue;
            if (validity_mask->at(nx, ny, 0) != 255) continue;

            return true;
        }
    }

    return false;
}

void
TextureAtlas::apply_edge_padding(void) {
    assert(image != NULL);
    assert(validity_mask != NULL);

    const int width = image->width();
    const int height = image->height();

    math::Matrix<float, 3, 3> gauss;
    gauss[0] = 1.0f; gauss[1] = 2.0f; gauss[2] = 1.0f;
    gauss[3] = 2.0f; gauss[4] = 4.0f; gauss[5] = 2.0f;
    gauss[6] = 1.0f; gauss[7] = 2.0f; gauss[8] = 1.0f;
    gauss /= 16.0f;

    /* Calculate the set of invalid pixels at the border of texture patches. */
    std::set<int> invalid_border_pixels;
    std::set<int>::iterator it = invalid_border_pixels.begin();
    for (int y = height - 1; 0 <= y; --y) {
        for (int x = width - 1; 0 <= x; --x) {
            if (validity_mask->at(x, y, 0) == 255) continue;
            if (!has_valid_neighbor(validity_mask, x, y)) continue;

            it = invalid_border_pixels.insert(it, y * width + x);
        }
    }

    /* Iteratively dilate border pixels until padding constants are reached. */
    for (unsigned int n = 0; n <= padding; ++n) {
        PixelVector new_valid_pixels;

        it = invalid_border_pixels.begin();
        for (;it != invalid_border_pixels.end(); it++) {
            int x = *it % width;
            int y = *it / width;

            /* Calculate new pixel value. */
            float norm = 0.0f;
            math::Vec3f value(0.0f);
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    int nx = x + i;
                    int ny = y + j;

                    if (nx < 0 || width <= nx) continue;
                    if (ny < 0 || height <= ny) continue;
                    if (validity_mask->at(nx, ny, 0) != 255) continue;

                    float w = gauss[(j + 1) * 3 + (i + 1)];
                    norm += w;
                    value += math::Vec3f(&image->at(nx, ny, 0)) * w;
                }
            }

            if (norm <= 0.0f) continue;

            for (int c = 0; c < 3; ++c) {
                image->at(x, y, c) = value[c] / norm;
            }
            new_valid_pixels.push_back(std::make_pair(x, y));
        }

        /* Mark the new valid pixels valid in the validity mask. */
        for (std::size_t i = 0; i < new_valid_pixels.size(); ++i) {
             int x = new_valid_pixels[i].first;
             int y = new_valid_pixels[i].second;

             validity_mask->at(x, y, 0) = 255;
        }

        invalid_border_pixels.clear();
        it = invalid_border_pixels.begin();

        /* Calculate the set of invalid pixels at the border of the valid area. */
        for (std::size_t i = 0; i < new_valid_pixels.size(); ++i) {
            int x = new_valid_pixels[i].first;
            int y = new_valid_pixels[i].second;

            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    int nx = x - i;
                    int ny = y - j;

                    if (nx < 0 || width <= nx) continue;
                    if (ny < 0 || height <= ny) continue;
                    if (validity_mask->at(nx, ny, 0) == 255) continue;

                    it = invalid_border_pixels.insert(it, ny * width + nx);
                }
            }
        }
    }
}

struct VectorCompare {
    bool operator()(math::Vec2f const & lhs, math::Vec2f const & rhs) {
        return lhs[0] < rhs[0] || (lhs[0] == rhs[0] && lhs[1] < rhs[1]);
    }
};

typedef std::map<math::Vec2f, std::size_t, VectorCompare> TexcoordMap;

void
TextureAtlas::merge_texcoords() {
    Texcoords tmp; tmp.swap(this->texcoords);

    TexcoordMap texcoord_map;
    for (math::Vec2f const & texcoord : tmp) {
        TexcoordMap::iterator iter = texcoord_map.find(texcoord);
        if (iter == texcoord_map.end()) {
            std::size_t texcoord_id = this->texcoords.size();
            texcoord_map[texcoord] = texcoord_id;
            this->texcoords.push_back(texcoord);
            this->texcoord_ids.push_back(texcoord_id);
        } else {
            this->texcoord_ids.push_back(iter->second);
        }
    }

}

void
TextureAtlas::finalize() {
    if (finalized) {
        throw util::Exception("TextureAtlas already finalized");
    }

    this->bin.reset();
    this->apply_edge_padding();
    this->validity_mask.reset();
    this->merge_texcoords();

    this->finalized = true;
}
