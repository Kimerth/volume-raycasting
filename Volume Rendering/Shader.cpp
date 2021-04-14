#include "Shader.h"
#include <glm/gtc/type_ptr.hpp>
#include "utils.h"
#include <iostream>
#include <vector>

Shader::Shader()
	:shader_programme(NULL)
{

}

void Shader::load(const char* vertexPath, const char* fragmentPath)
{
	std::string vstext = textFileRead(vertexPath);
	std::string fstext = textFileRead(fragmentPath);
	const char* vertex_shader = vstext.c_str();
	const char* fragment_shader = fstext.c_str();

	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, &vertex_shader, NULL);
	glCompileShader(vs);
	if (hasErrors(vs))
		exit(EXIT_FAILURE);

	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, &fragment_shader, NULL);
	glCompileShader(fs);
	if (hasErrors(fs))
		exit(EXIT_FAILURE);

	shader_programme = glCreateProgram();
	glAttachShader(shader_programme, fs);
	glAttachShader(shader_programme, vs);
}

void Shader::use() const
{
	glUseProgram(shader_programme);
}

void Shader::setFloat(const std::string& name, float value) const
{
	glUniform1f(glGetUniformLocation(shader_programme, name.c_str()), value);
}

void Shader::setVec2(const std::string& name, float x, float y) const
{
	glUniform2f(glGetUniformLocation(shader_programme, name.c_str()), x, y);
}

void Shader::setVec3(const std::string& name, const glm::vec3& value) const
{
	glUniform3fv(glGetUniformLocation(shader_programme, name.c_str()), 1, &value[0]);
}

void Shader::setVec3(const std::string& name, float x, float y, float z) const
{
	glUniform3f(glGetUniformLocation(shader_programme, name.c_str()), x, y, z);
}

void Shader::setMat4(const std::string& name, const glm::mat4& mat) const
{
	glUniformMatrix4fv(glGetUniformLocation(shader_programme, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
}

bool Shader::hasErrors(GLuint shader)
{
	GLint isCompiled = 0;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);
	if (isCompiled == GL_FALSE)
	{
		GLint maxLength = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);

		std::vector<GLchar> errorLog(maxLength);
		glGetShaderInfoLog(shader, maxLength, &maxLength, &errorLog[0]);

		std::cerr << errorLog.data() << std::endl;

		glDeleteShader(shader);
		return true;
	}

	return false;
}
