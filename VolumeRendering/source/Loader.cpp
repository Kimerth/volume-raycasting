#include "Loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

uchar* readNRRD(const char* path, int& width, int& height, int& depth, float& scaleX, float& scaleY, float& scaleZ)
{
    std::ifstream file(path, std::ios::binary);

    if (!file.good()) return nullptr;

    int sizeData = 0;
    scaleX = 1, scaleY = 1, scaleZ = 1;
    for (std::string line; std::getline(file, line) && !line.empty(); )
    {
        if (line.compare(0, 6, "sizes:") == 0)
        {
            std::istringstream iss(line.substr(6));

            iss >> width >> height >> depth;
            sizeData = width * height * depth;
        }
        else if (line.compare(0, 9, "spacings:") == 0)
        {
            std::istringstream iss(line.substr(9));

            iss >> scaleX >> scaleY>> scaleZ;
        }
    }

    uchar* buffer = new uchar[sizeData];
    file.read(reinterpret_cast<char*>(buffer), sizeData);

    file.close();

    return buffer;
}

uchar* readRAWfile(std::ifstream &file, uint &bytes)
{
    uint pos = file.tellg();
    file.seekg(0, file.end);
    int length = file.tellg();
    file.seekg(0, file.beg);
    file.ignore(std::numeric_limits<int>::max(), '\n');

    uchar* data = new uchar[length];
    bytes = 0;

    for(bytes = 0; file; bytes += DDS_BLOCKSIZE)
        file.read(reinterpret_cast<char*>(data + bytes), DDS_BLOCKSIZE);

    if (bytes == 0)
        exit(EXIT_FAILURE);

    return data;
}

#pragma region DDS

constexpr char DDS_ID[] = "DDS v3d";
constexpr char DDS_ID2[] = "DDS v3e";
const int headerLen = strlen(DDS_ID);

uint DDS_cachepos;

uint DDS_buffer;
uint DDS_bufsize;

unsigned short int DDS_INTEL = 1;

void DDS_initbuffer()
{
    DDS_buffer = 0;
    DDS_bufsize = 0;
}

inline uint DDS_shiftl(const uint value, const uint bits)
{
    return (bits >= 32) ? 0 : value << bits;
}

inline uint DDS_shiftr(const uint value, const uint bits)
{
    return (bits >= 32) ? 0 : value >> bits;
}

inline void DDS_swapuint(uint* x)
{
    uint tmp = *x;

    *x = ((tmp & 0xff) << 24) |
        ((tmp & 0xff00) << 8) |
        ((tmp & 0xff0000) >> 8) |
        ((tmp & 0xff000000) >> 24);
}

uint DDS_readbits(uint bits, uchar* data, size_t size)
{
    uint value;

    if (bits < DDS_bufsize)
    {
        DDS_bufsize -= bits;
        value = DDS_shiftr(DDS_buffer, DDS_bufsize);
    }
    else
    {
        value = DDS_shiftl(DDS_buffer, bits - DDS_bufsize);

        if (DDS_cachepos >= size) DDS_buffer = 0;
        else
        {
            DDS_buffer = *((uint*)&data[DDS_cachepos]);
            if (DDS_ISINTEL) DDS_swapuint(&DDS_buffer);
            DDS_cachepos += 4;
        }

        DDS_bufsize += 32 - bits;
        value |= DDS_shiftr(DDS_buffer, DDS_bufsize);
    }

    DDS_buffer &= DDS_shiftl(1, DDS_bufsize) - 1;

    return value;
}

void DDS_deinterleave(uchar* data, uint bytes, uint skip, uint block = 0, bool restore = false)
{
    uint i, j, k;

    uchar* data2, * ptr;

    if (skip <= 1) return;

    if (block == 0)
    {
        data2 = new uchar[bytes];

        if (!restore)
            for (ptr = data2, i = 0; i < skip; i++)
                for (j = i; j < bytes; j += skip) *ptr++ = data[j];
        else
            for (ptr = data, i = 0; i < skip; i++)
                for (j = i; j < bytes; j += skip) data2[j] = *ptr++;

        std::copy(data2, data2 + bytes, data);
    }
    else
    {
        data2 = new uchar[bytes < skip* block ? bytes : skip * block];

        if (!restore)
        {
            for (k = 0; k < bytes / skip / block; k++)
            {
                for (ptr = data2, i = 0; i < skip; i++)
                    for (j = i; j < skip * block; j += skip) *ptr++ = data[k * skip * block + j];

                std::copy(data2, data2 + skip * block, data + k * skip * block);
            }

            for (ptr = data2, i = 0; i < skip; i++)
                for (j = i; j < bytes - k * skip * block; j += skip) *ptr++ = data[k * skip * block + j];

            std::copy(data2, data2 + bytes - k * skip * block, data + k * skip * block);
        }
        else
        {
            for (k = 0; k < bytes / skip / block; k++)
            {
                for (ptr = data + k * skip * block, i = 0; i < skip; i++)
                    for (j = i; j < skip * block; j += skip) data2[j] = *ptr++;

                std::copy(data2, data2 + skip * block, data + k * skip * block);
            }

            for (ptr = data + k * skip * block, i = 0; i < skip; i++)
                for (j = i; j < bytes - k * skip * block; j += skip) data2[j] = *ptr++;

            std::copy(data2, data2 + bytes - k * skip * block, data + k * skip * block);
        }
    }

    delete[] data2;
}

void DDS_decode(uchar* chunk, uint size,
    uchar** data, uint &bytes,
    uint block)
{
    uint skip, strip;

    uchar* ptr1, * ptr2;

    uint cnt1, cnt2;
    int bits, act;

    DDS_initbuffer();

    DDS_cachepos = 0;
    
    auto readbits = [&](uint bits) 
    {
        return DDS_readbits(bits, chunk, size);
    };

    skip = readbits(2) + 1;
    strip = readbits(16) + 1;

    ptr1 = ptr2 = new uchar[3 * size];
    bytes = act = 0;

    while ((cnt1 = readbits(DDS_RL)) != 0)
    {
        bits = readbits(3);
        bits = bits >= 1 ? bits + 1 : bits;

        for (cnt2 = 0; cnt2 < cnt1; cnt2++)
        {
            if (strip == 1 || bytes <= strip) act += readbits(bits) - (1 << bits) / 2;
            else act += *(ptr2 - strip) - *(ptr2 - strip - 1) + readbits(bits) - (1 << bits) / 2;

            while (act < 0) act += 256;
            while (act > 255) act -= 256;

            if ((bytes & (DDS_BLOCKSIZE - 1)) == 0)
                ptr2 = &ptr1[bytes];

            *ptr2++ = act;
            bytes++;

            if (bytes % size == 0)
            {
                ptr2 = new uchar[bytes + size];
                std::copy(ptr1, ptr1 + bytes, ptr2);
                delete[] ptr1;
                ptr1 = ptr2;
                ptr2 = ptr2 + bytes;
            }
        }
    }

    *data = new uchar[bytes];
    std::copy(ptr1, ptr1 + bytes, *data);
    delete[] ptr1;

    DDS_deinterleave(*data, bytes, skip, block, true);
}

uchar* readDDSfile(std::ifstream& file, uint &bytes, uint block)
{
    int cnt;

    uchar* chunk, * data;
    uint size;

    if ((chunk = readRAWfile(file, size)) == NULL)
        exit(EXIT_FAILURE);

    DDS_decode(chunk, size, &data, bytes, block);

    delete[] chunk;

    return data;
}

#pragma endregion

uchar* readPVM(const char* path, int& width, int& height, int& depth, float& scaleX, float& scaleY, float& scaleZ)
{
    uchar* data;
    char* ptr;
    uint bytes, numc, block;

    bool is_DDS = false;

    scaleX = 1.0f, scaleY = 1.0f, scaleZ = 1.0f;

    std::ifstream file(path, std::ios::binary);
    if (!file.good())
        exit(EXIT_FAILURE);

    std::string header;
    std::getline(file, header);
    if (strcmp(header.c_str(), DDS_ID) == 0 || strcmp(header.c_str(), DDS_ID2) == 0)
        block = DDS_INTERLEAVE, is_DDS = true;

    if (is_DDS)
        data = readDDSfile(file, bytes, block);
    else
        data = readRAWfile(file, bytes);

    if (bytes < 5)
        return nullptr;

    ptr = strchr((char*)data, '\n') + 1;;

    if (sscanf_s((char*)ptr, "%d %d %d\n%g %g %g\n", &width, &height, &depth, &scaleX, &scaleY, &scaleZ) != 6)
        exit(EXIT_FAILURE);
    if (width < 1 || height < 1 || depth < 1 || scaleX <= 0.0f || scaleY <= 0.0f || scaleZ <= 0.0f)
        exit(EXIT_FAILURE);
    ptr = strchr((char*)ptr, '\n') + 1;

    ptr = strchr((char*)ptr, '\n') + 1;
    if (sscanf_s((char*)ptr, "%d\n", &numc) != 1) exit(EXIT_FAILURE);
    if (numc < 1) exit(EXIT_FAILURE);

    if (numc != 1)
        exit(EXIT_FAILURE);

    ptr = strchr((char*)ptr, '\n') + 1;

    uchar* volume = new uchar[width * height * depth * numc];

    std::copy(ptr, ptr + width * height * depth * numc, volume);
    free(data);

    return volume;
}

Format getFileFormat(const char* path)
{
    char* s = new char[strlen(path) + 1];
    memcpy(s, path, strlen(path) + 1);

    char* last_tok = s;
    for (char* tok = strchr(s, '.'); tok; last_tok = tok, tok = strchr(tok + 1, '.'));
    last_tok++;

    if (strcmp(last_tok, "nrrd") == 0)
        return Format::NRRD;
    else if (strcmp(last_tok, "pvm") == 0)
        return Format::PVM;
    else
        return Format::UNKNOWN;
}

uchar* readVolume(const char* path, int& width, int& height, int& depth, float& scaleX, float& scaleY, float& scaleZ)
{
    switch (getFileFormat(path))
    {
    case Format::NRRD:
        return readNRRD(path, width, height, depth, scaleX, scaleY, scaleZ);
        break;
    case Format::PVM:
        return readPVM(path, width, height, depth, scaleX, scaleY, scaleZ);
        break;
    default:
        break;
    }
}

float* readTF(const char* path)
{
    float* data = new float[4 * 256];

    std::ifstream file(path, std::ios::binary);
    if (!file.good())
        exit(EXIT_FAILURE);

    file.seekg(0, file.end);
    int length = file.tellg();
    file.seekg(0, file.beg);

    char* buffer = new char[length];
    file.read(buffer, length);

    char* ptr = strstr(buffer, "re=");
    for (int i = 0; ptr != NULL && i < 4 * 256; i+=4, ptr = strstr(ptr + 1, "re="))
    {
        float re, ge, be;
        float ra, ga, ba;
        sscanf_s(ptr,
            "re=%f\n"
            "ge=%f\n"
            "be=%f\n"
            "ra=%f\n"
            "ga=%f\n"
            "ba=%f\n", &re, &ge, &be, &ra, &ga, &ba);
        data[i]     = re;
        data[i + 1] = ge;
        data[i + 2] = be;
        data[i + 3] = (ra + ga + ba) / 3;
    }

    return data;
}
