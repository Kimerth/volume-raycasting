#pragma once
#include <string>
#include <glm/glm.hpp>

class Shader
{
public:
	Shader(const char* vertexPath, const char* fragmentPath);

	void use() const;

	void setFloat(const std::string& name, float value) const;

	void setVec3(const std::string& name, const glm::vec3& value) const;
	void setVec3(const std::string& name, float x, float y, float z) const;

	void setMat4(const std::string& name, const glm::mat4& mat) const;

private:
	GLuint shader_programme
};
