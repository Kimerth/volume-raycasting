#include "Shader.h"
#include <GL/glew.h>
#include <glm/gtc/type_ptr.hpp>
#include "utils.h"

Shader::Shader(const char* vertexPath, const char* fragmentPath)
{
	std::string vstext = textFileRead(vertexPath);
	std::string fstext = textFileRead(fragmentPath);
	const char* vertex_shader = vstext.c_str();
	const char* fragment_shader = fstext.c_str();

	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, &vertex_shader, NULL);
	glCompileShader(vs);
	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, &fragment_shader, NULL);
	glCompileShader(fs);

	shader_programme = glCreateProgram();
	glAttachShader(shader_programme, fs);
	glAttachShader(shader_programme, vs);
}

inline void Shader::use() const
{
	glUseProgram(shader_programme);
}

inline void Shader::setFloat(const std::string& name, float value) const
{
	glUniform1f(glGetUniformLocation(shader_programme, name.c_str()), value);
}

inline void Shader::setVec3(const std::string& name, const glm::vec3& value) const
{
	glUniform3fv(glGetUniformLocation(shader_programme, name.c_str()), 1, &value[0]);
}

inline void Shader::setVec3(const std::string& name, float x, float y, float z) const
{
	glUniform3f(glGetUniformLocation(shader_programme, name.c_str()), x, y, z);
}

inline void Shader::setMat4(const std::string& name, const glm::mat4& mat) const
{
	glUniformMatrix4fv(glGetUniformLocation(shader_programme, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
}
