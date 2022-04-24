#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "stb_image.h"
#include <thread>

#include <iostream>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080 - (25 + 40);

const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"uniform mat4 model; \n"
"uniform mat4 world; \n"
"uniform mat4 view; \n"
"uniform mat4 projection; \n"
"uniform float time; \n"
"void main()\n"
"{\n"
"   gl_Position = projection * world * view * model * vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"}\0";
const char* fragmentShaderSource = "#version 330 core\n"
"uniform vec4 col;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"   FragColor = col;\n"
"}\n\0";

class vec3 {
public:
    float x = 0;
    float y = 0;
    float z = 0;

    vec3() {}

    vec3(float x_, float y_, float z_) {
        x = x_;
        y = y_;
        z = z_;
    }

    void set(float x_, float y_, float z_) {
        x = x_;
        y = y_;
        z = z_;
    }
}

class octree {
public:
    float edge; // edge length
    float half_edge;
    float edge_sq;
    float mass = 0; // mass of all the particles inside this nodes's volume
    int level = 0; // number of divisions required to get to this point
    int population = 0;
    vec3* position; // position of the center of this node
    vec3* min_position = new vec3();
    vec3* max_position = new vec3();
    vec3* COM = new vec3(); // center of mass
    bool leaf = true; // whether not this node is a leaf node
    bool populated = false; // some nodes might be empty. distinction is important
    int* indices = nullptr; // array of the indices of the particles inside this node
    octree* children[8] = {};
    octree(vec3* pos, float edge_) {
        position = pos;
        edge = edge_;
        half_edge = edge / 2.0;
        edge_sq = edge_ * edge_;
        min_position->set(pos->x - half_edge, pos->y - half_edge, pos->z - half_edge);
        max_position->set(pos->x + half_edge, pos->y + half_edge, pos->z + half_edge);
    }

    ~octree() {
        // destructor
        delete position;
        delete indices;
        delete COM;
        delete min_position;
        delete max_position;
    }

    void decimate() {
        // call the destructor on all child nodes to clear memory
        if (!leaf) {
            for (int i = 0; i < 8; i ++) {
                children[i]->decimate();
                delete children[i];
            }
        }
    }
    
    void check(float vertices[], int indexes[], int array_length, octree* leaf_array[], int* leaf_index) {
        // temporary array to store the indices of the points inside this node
        int* arr = new int[array_length];

        // loop through all the indices of the parent node and check which
        // ones are inside the volume of this node.

        for (int j = 0; j < array_length; j++) {
            int i = indexes[j];
            if (vertices[i] < max_position->x && vertices[i] > min_position->x &&
                vertices[i + 1] < max_position->y && vertices[i + 1] > min_position->y &&
                vertices[i + 2] < max_position->z && vertices[i + 2] > min_position->z) {
                arr[population] = i;
                population++;
                mass += 7;
                populated = true;
            }
        }

        // the array that was created to store the new indices is as long
        // as the source array, it is unlikely that the new array will
        // contain all the points it's parent node did, so for the sake of
        // efficiency, I am resizing the array to be as small as possible.

        if (population > 0) {
            indices = new int[population];
            memcpy(indices, arr, population * sizeof(int));
            float ax = 0;
            float ay = 0;
            float az = 0;
            // iterate through all point within this node and compute C.O.M
            for (int i = 0; i < population; i++) {
                int index = indices[i];

                ax += vertices[index];
                ay += vertices[index + 1];
                az += vertices[index + 2];
            }

            COM->set(ax / population, ay / population, az / population);
        }

        if (population > 1 && level < 22) {
            divide(vertices, leaf_array, leaf_index);
            leaf = false;
        }
        
        delete[] arr;

        if (population == 1) {
            leaf_array[*leaf_index] = this;
            *leaf_index += 1;
        }
    }

    void divide(float verts[], octree* leaf_array[], int* leaf_index) {
        float quater_edge = edge / 4.0;
        children[0] = new octree(new vec3(position->x + quater_edge, position->y + quater_edge, position->z + quater_edge), half_edge);
        children[1] = new octree(new vec3(position->x + quater_edge, position->y + quater_edge, position->z - quater_edge), half_edge);
        children[2] = new octree(new vec3(position->x + quater_edge, position->y - quater_edge, position->z - quater_edge), half_edge);
        children[3] = new octree(new vec3(position->x + quater_edge, position->y - quater_edge, position->z + quater_edge), half_edge);
        children[4] = new octree(new vec3(position->x - quater_edge, position->y + quater_edge, position->z + quater_edge), half_edge);
        children[5] = new octree(new vec3(position->x - quater_edge, position->y + quater_edge, position->z - quater_edge), half_edge);
        children[6] = new octree(new vec3(position->x - quater_edge, position->y - quater_edge, position->z - quater_edge), half_edge);
        children[7] = new octree(new vec3(position->x - quater_edge, position->y - quater_edge, position->z + quater_edge), half_edge);
        for (int i = 0; i < 8; i++) {
            children[i]->level = level + 1;
            children[i]->check(verts, indices, population, leaf_array, leaf_index);
        }
    }

    void traverse(octree* target, float* x, float* y, float* z, float* ax, float* ay, float* az) {
        if (target != this) {
            if (!leaf) {
                for (int i = 0; i < 8; i++) {
                    octree* ctx = children[i];
                    if (!ctx->leaf) {
                        float distx = ctx->COM->x - *x;
                        float disty = ctx->COM->y - *y;
                        float distz = ctx->COM->z - *z;
                        float dist = (distx * distx + disty * disty + distz * distz);
                        if (ctx->edge_sq / dist < 1) {
                            float coefficient = (6.67408e-11 * ctx->mass / dist);
                            float x2 = coefficient * distx;
                            float y2 = coefficient * disty;
                            float z2 = coefficient * distz;
                            *ax += x2;
                            *ay += y2;
                            *az += z2;
                        }
                        else {
                            ctx->traverse(target, x, y, z, ax, ay, az);
                        }
                    }

                    else {
                        ctx->traverse(target, x, y, z, ax, ay, az);
                    }
                }
            }

            else {
                if (populated) {
                    float distx = COM->x - *x;
                    float disty = COM->y - *y;
                    float distz = COM->z - *z;
                    float dist = (distx * distx + disty * disty + distz * distz);
                    float coefficient = (6.67408e-11 * mass / dist);

                    float x2 = coefficient * distx;
                    float y2 = coefficient * disty;
                    float z2 = coefficient * distz;
                    *ax += x2;
                    *ay += y2;
                    *az += z2;
                }
            }
        }
    }
}

void propogate(int begin, int end, octree* root, octree* nodes[], float position[], float velocity[]) {
    for (int i = begin; i < end; i++) {
        int index = nodes[i]->indices[0];
        float* x = &nodes[i]->position->x;
        float* y = &nodes[i]->position->y;
        float* z = &nodes[i]->position->z;
        float* ax = new float(0);
        float* ay = new float(0);
        float* az = new float(0);
        root->traverse(nodes[i], x, y, z, ax, ay, az);
        velocity[index] += *ax;
        velocity[index + 1] += *ay;
        velocity[index + 2] += *az;
        position[index] += velocity[index];
        position[index + 1] += velocity[index + 1];
        position[index + 2] += velocity[index + 2];
        delete ax;
        delete ay;
        delete az;
    }
}

int main() {
    // glfw: initialize and configure
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // glfw window creation
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Logan's Godly code", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    unsigned int vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    unsigned int shaderProgram;
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // delete old shaders because once they are combined into a program we chillin
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // number of particles that will be simulated
    int NUM_PARTICLES = 10000;

    // postion array for each particle
    float* vertices;
    vertices = new float[NUM_PARTICLES * 3];

    // velocity array for each particle
    float* velocities;
    velocities = new float[NUM_PARTICLES * 3];

    // initialize the position and velocity vectors of every particle.
    for (int i = 0; i < NUM_PARTICLES * 3; i+=3) {
        float curve = 2 * 3.14159 * (rand() / float(RAND_MAX));
        float radius = 1 * (rand() / float(RAND_MAX));
        float x = (cos(curve) - sin(curve));
        float y = (cos(curve) + sin(curve));
        vertices[i] = radius * x; // 1.5 * ((rand() / float(RAND_MAX)) - 0.5);
        vertices[i + 1] = radius * y; // 1.5 * ((rand() / float(RAND_MAX)) - 0.5);
        vertices[i+2] =  0.02 * ((rand() / float(RAND_MAX)) - 0.5);   
        float vel = sqrt(6.67e-11 * NUM_PARTICLES * 200 / radius) * 0.05;

        velocities[i] = -y * vel; // 0.0 * ((rand() / float(RAND_MAX)) - 0.5);
        velocities[i + 1] = x * vel; // 0.0 * ((rand() / float(RAND_MAX)) - 0.5);
        velocities[i + 2] = 0.0; // 0.0 * ((rand() / float(RAND_MAX)) - 0.5);
    }

    // every time a new octree is created, it checks the list of particles
    // it's parent contained instead of all of them because a child node
    // cannot contain nodes that the parent did not. the root node does not
    // have a parent, so a list of indices must be provided.

    int* seed = new int[NUM_PARTICLES];
    for (int i = 0; i < NUM_PARTICLES; i++) {
        seed[i] = i * 3;
    }

    // very important. This array contains pointers to every single leaf node
    // only these nodes need to propogated.
    octree** leaf_node_array;
    leaf_node_array = new octree * [NUM_PARTICLES];

    int* leaf_index = new int(0);

    // the root node. contains all other nodes. should be big enough to contain
    // the whole simulation
    octree* root = new octree(new vec3(0, 0, 0), 10.0f);

    unsigned int VBO;
    unsigned int VAO;

    // vertex buffer
    glGenBuffers(1, &VBO);
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * NUM_PARTICLES * 3, vertices, GL_STATIC_DRAW);


    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // main loop
    while (!glfwWindowShouldClose(window)) {
        // input
        processInput(window);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * NUM_PARTICLES * 3, vertices, GL_STATIC_DRAW);
        
        float time = 1;// * glfwGetTime();

        *leaf_index = 0;

        // create a new octree and assign it as the root node
        root = new octree(new vec3(0.1, 0.0, 0.0), 10.0f);

        // construct the octree using data from the points.
        // this will recursively subdived the octree, making more
        // octrees until the simulated volume is divided into partitions
        // containing only 1 particle. There is a subdivision limit,
        // so larger sims may need more subdivisions

        root->check(vertices, seed, NUM_PARTICLES, leaf_node_array, leaf_index);
        
        // the number of particles each thread need to deal with.
        int step = int(*leaf_index  / 12);

        // propogate(0, *leaf_index, root, leaf_node_array, vertices, velocities);
        propogate(0, step-1, root, leaf_node_array, vertices, velocities);
        
        // create 11 threads and assign each one a portion of the particles
        // do deal with.     starting point:    ending point:
        //                           |               |
        //                           |               |
        //                           V               V
        std::thread t1( propogate, step,      step * 2 -  1, root, leaf_node_array, vertices, velocities);
        std::thread t2( propogate, step * 2,  step * 3 -  1, root, leaf_node_array, vertices, velocities);
        std::thread t3( propogate, step * 3,  step * 4 -  1, root, leaf_node_array, vertices, velocities);
        std::thread t4( propogate, step * 4,  step * 5 -  1, root, leaf_node_array, vertices, velocities);
        std::thread t5( propogate, step * 5,  step * 6 -  1, root, leaf_node_array, vertices, velocities);
        std::thread t6( propogate, step * 6,  step * 7 -  1, root, leaf_node_array, vertices, velocities);
        std::thread t7( propogate, step * 7,  step * 8 -  1, root, leaf_node_array, vertices, velocities);
        std::thread t8( propogate, step * 8,  step * 9 -  1, root, leaf_node_array, vertices, velocities);
        std::thread t9( propogate, step * 9,  step * 10 - 1, root, leaf_node_array, vertices, velocities);
        std::thread t10(propogate, step * 10, step * 11 - 1, root, leaf_node_array, vertices, velocities);
        std::thread t11(propogate, step * 11, step * 12, root, leaf_node_array, vertices, velocities);
        
        // wait for the threads to finish their work
        t1.join();
        t2.join();
        t3.join();
        t4.join();
        t5.join();
        t6.join();
        t7.join();
        t8.join();
        t9.join();
        t10.join();
        t11.join();
        
        // call the method than deletes all of the octree's branches
        root->decimate();
        // delete the octree itself
        delete root;

        // particle color
        float r = 1;//cos(time);
        float g = 0.3;//cos(time + 2 * 3.14 / 3);
        float b = 0.1;//cos(time + 4 * 3.14 / 3);

        // transformation matrices
        glm::mat4 trans = glm::mat4(1.0f);
        trans = glm::rotate(trans, (float)time, glm::vec3(0.0, 0.0, 1.0));
        trans = glm::scale(trans, glm::vec3(1.0, 0.5 * cos(time), 0.5));

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::rotate(model, glm::radians(10.0f * time), glm::vec3(1.0f, 1.0f, 1.0f));

        glm::mat4 world = glm::mat4(1.0f);
        world = glm::translate(world, glm::vec3(0.0f, 0.0f, -4.0f));

        glm::mat4 view = glm::mat4(1.0f);
        view = glm::translate(view, glm::vec3(0.0f, 0.0f, 0.0f));

        glm::mat4 projection;
        projection = glm::perspective(glm::radians(45.0f), float(SCR_WIDTH) / float(SCR_HEIGHT), 0.1f, 100.0f);

        // set the shader as the active shader
        glUseProgram(shaderProgram);

        // pass transformation matrices and other uniforms to the shader program
        int modelMatrix = glGetUniformLocation(shaderProgram, "model");
        glUniformMatrix4fv(modelMatrix, 1, GL_FALSE, glm::value_ptr(model));

        int worldMatrix = glGetUniformLocation(shaderProgram, "world");
        glUniformMatrix4fv(worldMatrix, 1, GL_FALSE, glm::value_ptr(world));

        int viewMatrix = glGetUniformLocation(shaderProgram, "view");
        glUniformMatrix4fv(viewMatrix, 1, GL_FALSE, glm::value_ptr(view));

        int projectionMatrix = glGetUniformLocation(shaderProgram, "projection");
        glUniformMatrix4fv(projectionMatrix, 1, GL_FALSE, glm::value_ptr(projection));
        
        int vertexColorLocation = glGetUniformLocation(shaderProgram, "col");
        glUniform4f(vertexColorLocation, r, g, b, 0.5f);

        int vertexTimeLocation = glGetUniformLocation(shaderProgram, "time");
        glUniform1f(vertexTimeLocation, time);

        // size of particles
        glPointSize(1);
        // additive blending, allowing for color variation in dense regions
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glEnable(GL_BLEND);

        glBindVertexArray(VAO);
        // render the scene
        glDrawArrays(GL_POINTS,0, NUM_PARTICLES);


        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

// glfw: whenever the window size changes
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    // make sure the viewport matches the new window dimensions
    glViewport(0, 0, width, height);
}
