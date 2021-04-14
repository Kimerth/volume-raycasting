#pragma once
#include <string>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>

class Shader
{
public:
	GLuint shader_programme;

	Shader();
	void load(const char* vertexPath, const char* fragmentPath);

	void use() const;

	void setFloat(const std::string& name, float value) const;

	void setVec2(const std::string& name, float x, float y) const;

	void setVec3(const std::string& name, const glm::vec3& value) const;
	void setVec3(const std::string& name, float x, float y, float z) const;

	void setMat4(const std::string& name, const glm::mat4& mat) const;

private:
	bool hasErrors(GLuint shader);
};
