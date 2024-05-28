



#if 0
void line(int x0, int y0, int x1, int y1, TGAImage& image, TGAColor color) {
    bool steep = false;
    if (std::abs(x0 - x1) < std::abs(y0 - y1)) {
        std::swap(x0, y0);
        std::swap(x1, y1);
        steep = true;
    }
    if (x0 > x1) {
        std::swap(x0, x1);
        std::swap(y0, y1);
    }
    int dx = x1 - x0;
    int dy = y1 - y0;
    float derror = std::abs(dy / float(dx));
    float error = 0;
    int y = y0;
    for (int x = x0; x <= x1; x++) {
        if (steep) {
            image.set(y, x, color);
            //image.flip_vertically();
            //image.write_tga_file((char)x + "output.tga");
        }
        else {
            image.set(x, y, color);
            //image.flip_vertically();
            //image.write_tga_file((char)x + "output.tga");
        }
        error += derror;
        if (error > .5) {
            y += (y1 > y0 ? 1 : -1);
            error -= 1.;
        }
    }
}

void scan(int x0, int x1, int y, TGAImage & image, TGAColor color) {
    for (int i = x0; i < x1; i++) {
        image.set(x0 + i, y, color);
    }
}

void triangle(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage& image, TGAColor color) {
    if (t0.y > t1.y) std::swap(t0, t1);
    if (t0.y > t2.y) std::swap(t0, t2);
    if (t1.y > t2.y) std::swap(t1, t2);

    int total_height = t2.y - t0.y;

    for (int y = t0.y; y <= t1.y; y++) {
        int segment_height = t1.y - t0.y + 1;
        float alpha = (float)(y - t0.y) / total_height;
        float beta = (float)(y - t0.y) / segment_height;
        Vec2i A = t0 + (t2 - t0) * alpha;
        Vec2i B = t0 + (t1 - t0) * beta;
        if (A.x > B.x) std::swap(A, B);
        
        //line(A.x, A.y, B.x, B.y, image, red);
        for (int j = A.x; j <= B.x; j++) {
            image.set(j, y, color);
        }
    }

}

void triangle2(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage& image, TGAColor color) {
    if (t0.y > t1.y) std::swap(t0, t1);
    if (t0.y > t2.y) std::swap(t0, t2);
    if (t1.y > t2.y) std::swap(t1, t2);

    int total_height = t2.y - t0.y;
    int i = 0;
    for (int y = t0.y; y <= t2.y; y++) {
        int segment_height = t1.y - t0.y + 1;
        float alpha = (float)(y - t0.y) / total_height;
        float beta = y <= t1.y ? (float)(y - t0.y) / segment_height : -1;
        float gama = y > t1.y ? (float)(y - t1.y) / (t2.y - t1.y) : -1;
        Vec2i A = t0 + (t2 - t0) * alpha;
        Vec2i B = beta != -1 ? t0 + (t1 - t0) * beta : t1 + (t2 - t1) * gama;
        if (A.x > B.x) std::swap(A, B);
        //line(A.x, y, B.x, y, image, red);
        for (int j = A.x; j <= B.x; j++) {
            image.set(j, y, color);
        }
        i++;
    }

}


/*
void triangle(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage& image, TGAColor color) {
    if (t0.y > t1.y) std::swap(t0, t1);
    if (t0.y > t2.y) std::swap(t0, t2);
    if (t1.y > t2.y) std::swap(t1, t2);

    bool enable = true;
    bool in_out = false;

    line(t0.x, t0.y, t1.x, t1.y, image, color);
    line(t1.x, t1.y, t2.x, t2.y, image, color);
    line(t0.x, t0.y, t2.x, t2.y, image, color);

    for (int i = t2.y - 1; i > t0.y + 1; i--) {
        for (int j = 0; j < image.get_width(); j++) {
            if (image.get(j, i).a != 0 || image.get(j, i).r != 0 || image.get(j, i).g != 0 || image.get(j, i).b != 0) {
                enable = (enable == true)&&(in_out == false) ? false : true;
                in_out = true;
            }
            if (in_out == true && !(image.get(j, i).a != 0 || image.get(j, i).r != 0 || image.get(j, i).g != 0 || image.get(j, i).b != 0)) {
                in_out = !in_out;
            }
            if (!enable) {
                image.set(j, i, color);
            }
        }
    }
}
*/


int main(int argc, char** argv) {
    TGAImage image(width, height, TGAImage::RGB);
    /*
    if (2 == argc) {
        model = new Model(argv[1]);
    }
    else {
        model = new Model("obj/african_head.obj");
    }


    for (int i = 0; i < model->nfaces(); i++) {
        std::vector<int> face = model->face(i);
        for (int j = 0; j < 3; j++) {
            Vec3f v0 = model->vert(face[j]);
            Vec3f v1 = model->vert(face[(j + 1) % 3]);
            int x0 = (v0.x + 1.) * width / 2.;
            int y0 = (v0.y + 1.) * height / 2.;
            int x1 = (v1.x + 1.) * width / 2.;
            int y1 = (v1.y + 1.) * height / 2.;
            line(x0, y0, x1, y1, image, white);
        }
    }
    */

    //Vec2i t0[3] = { Vec2i(10, 70),   Vec2i(50, 160),  Vec2i(70, 80) };
    //Vec2i t1[3] = { Vec2i(180, 50),  Vec2i(150, 1),   Vec2i(70, 180) };
    Vec2i t2[3] = { Vec2i(18, 15), Vec2i(100, 40), Vec2i(60, 100) };
    //triangle(t0[0], t0[1], t0[2], image, red);
    //triangle(t1[0], t1[1], t1[2], image, white);
    //triangle(t2[0], t2[1], t2[2], image, green);
    triangle2(t2[0], t2[1], t2[2], image, green);

    for (int i = 0; i < 100; i++) {
        //line(30, 30 + i, 300, 30 + i,image,red);
    }
    
    image.flip_vertically(); // i want to have the origin at the left bottom corner of the image
    image.write_tga_file("output.tga");
    delete model;
    return 0;
}
#endif

#if 0
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>
#include <sstream>
#include "tgaimage.h"
#include "model.h"
#include "geometry.h"

const int width = 800;
const int height = 800;
const int depth = 255;

Model* model = NULL;
int* zbuffer = NULL;
Vec3f light_dir = Vec3f(-1, -1, 1).normalize();
Vec3f eye(1, 1, 6);
Vec3f center(0, 0, 0);

Matrix viewport(int x, int y, int w, int h) {
    Matrix m = Matrix::identity(4);
    m[0][3] = x + w / 2.f;
    m[1][3] = y + h / 2.f;
    m[2][3] = depth / 2.f;

    m[0][0] = w / 2.f;
    m[1][1] = h / 2.f;
    m[2][2] = depth / 2.f;
    return m;
}

Matrix lookat(Vec3f eye, Vec3f center, Vec3f up) {
    Vec3f z = (eye - center).normalize();
    Vec3f x = (up ^ z).normalize();
    Vec3f y = (z ^ x).normalize();
    Matrix res = Matrix::identity(4);
    for (int i = 0; i < 3; i++) {
        res[0][i] = x[i];
        res[1][i] = y[i];
        res[2][i] = z[i];
        res[i][3] = -center[i];
    }
    return res;
}

void triangle(Vec3i t0, Vec3i t1, Vec3i t2, float ity0, float ity1, float ity2, TGAImage& image, int* zbuffer) {
    if (t0.y == t1.y && t0.y == t2.y) return; // i dont care about degenerate triangles
    if (t0.y > t1.y) { std::swap(t0, t1); std::swap(ity0, ity1); }
    if (t0.y > t2.y) { std::swap(t0, t2); std::swap(ity0, ity2); }
    if (t1.y > t2.y) { std::swap(t1, t2); std::swap(ity1, ity2); }

    int total_height = t2.y - t0.y;
    for (int i = 0; i < total_height; i++) {
        bool second_half = i > t1.y - t0.y || t1.y == t0.y;
        int segment_height = second_half ? t2.y - t1.y : t1.y - t0.y;
        float alpha = (float)i / total_height;
        float beta = (float)(i - (second_half ? t1.y - t0.y : 0)) / segment_height; // be careful: with above conditions no division by zero here
        Vec3i A = t0 + Vec3f(t2 - t0) * alpha;
        Vec3i B = second_half ? t1 + Vec3f(t2 - t1) * beta : t0 + Vec3f(t1 - t0) * beta;
        float ityA = ity0 + (ity2 - ity0) * alpha;
        float ityB = second_half ? ity1 + (ity2 - ity1) * beta : ity0 + (ity1 - ity0) * beta;
        if (A.x > B.x) { std::swap(A, B); std::swap(ityA, ityB); }
        for (int j = A.x; j <= B.x; j++) {
            float phi = B.x == A.x ? 1. : (float)(j - A.x) / (B.x - A.x);
            Vec3i    P = Vec3f(A) + Vec3f(B - A) * phi;
            float ityP = ityA + (ityB - ityA) * phi;
            int idx = P.x + P.y * width;
            if (P.x >= width || P.y >= height || P.x < 0 || P.y < 0) continue;
            if (zbuffer[idx] < P.z) {
                zbuffer[idx] = P.z;
                image.set(P.x, P.y, TGAColor(255, 255, 255) * ityP);
            }
        }
    }
}

int main(int argc, char** argv) {
    if (2 == argc) {
        model = new Model(argv[1]);
    }
    else {
        model = new Model("obj/african_head.obj");
    }

    zbuffer = new int[width * height];
    for (int i = 0; i < width * height; i++) {
        zbuffer[i] = std::numeric_limits<int>::min();
    }

    //debug info
    std::fstream  debug;
    debug.open("debug.txt", std::ios::out | std::ios::app);

    { // draw the model
        Matrix ModelView = lookat(eye, center, Vec3f(0, 1, 0));
        Matrix Projection = Matrix::identity(4);
        Matrix ViewPort = viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
        Projection[3][2] = -1.f / (eye - center).norm();

        std::cerr << ModelView << std::endl;
        std::cerr << Projection << std::endl;
        std::cerr << ViewPort << std::endl;
        Matrix z = (ViewPort * Projection * ModelView);
        std::cerr << z << std::endl;

        TGAImage image(width, height, TGAImage::RGB);
        for (int i = 0; i < model->nfaces(); i++) {
            std::vector<int> face = model->face(i);
            Vec3i screen_coords[3];
            Vec3f world_coords[3];
            float intensity[3];
            for (int j = 0; j < 3; j++) {
                Vec3f v = model->vert(face[j]);
                screen_coords[j] = Vec3f(z * Matrix(v));
                world_coords[j] = v;
                intensity[j] = model->norm(i, j) * light_dir;
                if (intensity[j] < 30 || 1){
                    debug << intensity[j] << " ";
                }
            }
            //debug << std::endl;
            triangle(screen_coords[0], screen_coords[1], screen_coords[2], intensity[0], intensity[1], intensity[2], image, zbuffer);
        }
        image.flip_vertically(); // i want to have the origin at the left bottom corner of the image
        image.write_tga_file("output.tga");
    }

    { // dump z-buffer (debugging purposes only)
        TGAImage zbimage(width, height, TGAImage::GRAYSCALE);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                zbimage.set(i, j, TGAColor(zbuffer[i + j * width]));
            }
        }
        zbimage.flip_vertically(); // i want to have the origin at the left bottom corner of the image
        zbimage.write_tga_file("zbuffer.tga");
    }
    delete model;
    delete[] zbuffer;
    return 0;
}
#endif

#include <vector>
#include <iostream>

#include "tgaimage.h"
#include "model.h"
#include "geometry.h"
#include "our_gl.h"

Model *model     = NULL;
const int width  = 800;
const int height = 800;

Vec3f light_dir(1, 1, 1);
Vec3f       eye(0, -1, 3);
Vec3f    center(0, 0, 0);
Vec3f        up(0, 1, 0);

struct GouraudShader : public IShader {
    Vec3f varying_intensity; // written by vertex shader, read by fragment shader
    mat<2, 3, float> varying_uv;        // same as above
    virtual Vec4f vertex(int iface, int nthvert) {
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));
        varying_intensity[nthvert] = std::max(0.f, model->normal(iface, nthvert)*light_dir); // get diffuse lighting intensity
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert)); // read the vertex from .obj file
        return Viewport*Projection*ModelView*gl_Vertex; // transform it to screen coordinates
    }

    virtual bool fragment(Vec3f bar, TGAColor &color) {
        float intensity = varying_intensity*bar;   // interpolate intensity for the current pixel
        Vec2f uv = varying_uv * bar;                 // interpolate uv for the current pixel
        color = model->diffuse(uv) * intensity; // well duh
        //model->diffuse(uv).color_print();
        
        return false;                              // no, we do not discard this pixel
    }
};

int main(int argc, char** argv) {
    if (2==argc) {
        model = new Model(argv[1]);
    } else {
        model = new Model("obj/african_head/african_head.obj");
    }

    lookat(eye, center, up);
    viewport(width/8, height/8, width*3/4, height*3/4);
    projection(-1.f/(eye-center).norm());
    light_dir.normalize();

    TGAImage image  (width, height, TGAImage::RGB);
    TGAImage zbuffer(width, height, TGAImage::GRAYSCALE);

    GouraudShader shader;
    for (int i=0; i<model->nfaces(); i++) {
        Vec4f screen_coords[3];
        for (int j=0; j<3; j++) {
            screen_coords[j] = shader.vertex(i, j);
        }
        triangle(screen_coords, shader, image, zbuffer);
    }

    image.  flip_vertically(); // to place the origin in the bottom left corner of the image
    zbuffer.flip_vertically();
    image.  write_tga_file("output.tga");
    zbuffer.write_tga_file("zbuffer.tga");

    delete model;
    return 0;
}