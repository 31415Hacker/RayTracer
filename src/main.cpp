<<<<<<< HEAD
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

// Vertex shader source
const char* vertShaderSrc = R"(
#version 330 core
layout(location = 0) in vec2 pos;
void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
}
)";

// Fragment shader source (outputs red)
const char* fragShaderSrc = R"(
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(1, 0, 0, 1); // Red
}
)";

int main() {
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(640, 480, "Red Window", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { std::cerr << "GLEW error\n"; return -1; }

    // Compile vertex shader
    GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertShader, 1, &vertShaderSrc, nullptr);
    glCompileShader(vertShader);

    // Compile fragment shader
    GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragShader, 1, &fragShaderSrc, nullptr);
    glCompileShader(fragShader);

    // Link shaders into a program
    GLuint shaderProg = glCreateProgram();
    glAttachShader(shaderProg, vertShader);
    glAttachShader(shaderProg, fragShader);
    glLinkProgram(shaderProg);

    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    // Fullscreen quad (two triangles)
    float quad[] = {
        -1, -1,  1, -1,  1, 1,
        -1, -1,  1,  1, -1, 1
    };
    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProg);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
=======
// src/main.cpp

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iterator>

// GLEW / GLFW / GLM
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// TinyOBJLoader
#define TINYOBJLOADER_IMPLEMENTATION
#include "/home/aryanshandanu18/3D rendering project/tiny_obj_loader.h"

#include "shader.h"

#define CHECK_GL_ERROR()                                   \
    {                                                      \
        GLenum err;                                        \
        while ((err = glGetError()) != GL_NO_ERROR)        \
        {                                                  \
            std::cerr << "GL ERROR: " << err << std::endl; \
        }                                                  \
    }

std::string readFile(const std::string &filepath)
{
    std::ifstream file(filepath);

    if (!file)
    {
        std::cerr << "Error opening file!" << std::endl;
        return "";
    }

    std::ostringstream ss;
    ss << file.rdbuf(); // read entire file buffer into stringstream

    return ss.str(); // get std::string
}

int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(800, 600, "Raytracer", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    glewInit();

    glViewport(0, 0, 800, 600);

    Shader shader3D("shaders/shader.vert", "shaders/shader.frag");

    // Now we will draw 2 triangles (quad) using VBO only:
    // 6 vertices for 2 triangles:
    float quadVertices[] = {
        // First triangle
        1.0f, 1.0f,
        1.0f, -1.0f,
        -1.0f, -1.0f,
        // Second triangle
        -1.0f, -1.0f,
        -1.0f, 1.0f,
        1.0f, 1.0f};

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    // Bind VAO first
    glBindVertexArray(VAO);

    // VBO
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    // Tell OpenGL how to interpret vertex data
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // Unbind (optional for safety)
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    // Optional: set clear color
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw quad-buffer
        shader3D.use();
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glDrawArrays(GL_TRIANGLES, 0, 6); // 6 vertices = 2 triangles = quad
        CHECK_GL_ERROR();

>>>>>>> d5a2be9 (soft shadows and adjustable full model matrix)
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

<<<<<<< HEAD
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProg);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
=======
    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    glfwDestroyWindow(window);
    glfwTerminate();
>>>>>>> d5a2be9 (soft shadows and adjustable full model matrix)
}