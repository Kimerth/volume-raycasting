#include "Loader.h"
#include <fstream>
#include <sstream>
#include <iostream>

uchar* readNRRD(const char* path, int& width, int& height, int& depth)
{
    std::ifstream file(path, std::ios::binary);

    if (!file.good()) return nullptr;

    int sizeData = 0;
    for (std::string line; std::getline(file, line) && !line.empty(); )
    {
        if (line.compare(0, 6, "sizes:") == 0)
        {
            std::istringstream iss(line.substr(6));

            iss >> width >> height >> depth;
            sizeData = width * height * depth;
        }
    }

    uchar* buffer = new uchar[sizeData];
    file.read(reinterpret_cast<char*>(buffer), sizeData);

    file.close();

    return buffer;
}

uchar* readDDSfile(std::ifstream &file, uint* bytes);
void DDS_decode(uchar* chunk, uint size,
    uchar** data, uint* bytes,
    uint block = 0);
uchar* readRAWfile(std::ifstream &file, uint* bytes);

uchar* readRAWfile(std::ifstream &file, uint* bytes)
{
    uint cnt;

    uint pos = file.tellg();
    file.seekg(0, file.end);
    int length = file.tellg();
    file.seekg(0, file.beg);
    file.ignore(std::numeric_limits<int>::max(), '\n');

    uchar* data = new uchar[length];
    cnt = 0;

    do
    {
        file.read(reinterpret_cast<char*>(data + cnt), DDS_BLOCKSIZE);
        cnt += DDS_BLOCKSIZE;
    } while (file);

    if (cnt == 0)
    {
        free(data);
        return(NULL);
    }

    if ((data = (uchar*)realloc(data, cnt)) == NULL) 
        exit(EXIT_FAILURE);

    *bytes = cnt;

    return(data);
}

#pragma region DDS

constexpr char DDS_ID[] = "DDS v3d";
constexpr char DDS_ID2[] = "DDS v3e";
const int headerLen = strlen(DDS_ID);

uchar* DDS_cache;
uint DDS_cachepos, DDS_cachesize;

uint DDS_buffer;
uint DDS_bufsize;

unsigned short int DDS_INTEL = 1;

void DDS_initbuffer()
{
    DDS_buffer = 0;
    DDS_bufsize = 0;
}

inline void DDS_clearbits()
{
    DDS_cache = NULL;
    DDS_cachepos = 0;
    DDS_cachesize = 0;
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

inline uint DDS_readbits(uint bits)
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

        if (DDS_cachepos >= DDS_cachesize) DDS_buffer = 0;
        else
        {
            DDS_buffer = *((uint*)&DDS_cache[DDS_cachepos]);
            if (DDS_ISINTEL) DDS_swapuint(&DDS_buffer);
            DDS_cachepos += 4;
        }

        DDS_bufsize += 32 - bits;
        value |= DDS_shiftr(DDS_buffer, DDS_bufsize);
    }

    DDS_buffer &= DDS_shiftl(1, DDS_bufsize) - 1;

    return(value);
}

inline void DDS_loadbits(uchar* data, uint size)
{
    DDS_cache = data;
    DDS_cachesize = size;

    if ((DDS_cache = (uchar*)realloc(DDS_cache, DDS_cachesize + 4)) == NULL) 
        exit(EXIT_FAILURE);
    *((uint*)&DDS_cache[DDS_cachesize]) = 0;

    DDS_cachesize = 4 * ((DDS_cachesize + 3) / 4);
    if ((DDS_cache = (uchar*)realloc(DDS_cache, DDS_cachesize)) == NULL) 
        exit(EXIT_FAILURE);
}

void DDS_deinterleave(uchar* data, uint bytes, uint skip, uint block = 0, bool restore = false)
{
    uint i, j, k;

    uchar* data2, * ptr;

    if (skip <= 1) return;

    if (block == 0)
    {
        if ((data2 = (uchar*)malloc(bytes)) == NULL) 
            exit(EXIT_FAILURE);

        if (!restore)
            for (ptr = data2, i = 0; i < skip; i++)
                for (j = i; j < bytes; j += skip) *ptr++ = data[j];
        else
            for (ptr = data, i = 0; i < skip; i++)
                for (j = i; j < bytes; j += skip) data2[j] = *ptr++;

        memcpy(data, data2, bytes);
    }
    else
    {
        if ((data2 = (uchar*)malloc((bytes < skip * block) ? bytes : skip * block)) == NULL) 
            exit(EXIT_FAILURE);

        if (!restore)
        {
            for (k = 0; k < bytes / skip / block; k++)
            {
                for (ptr = data2, i = 0; i < skip; i++)
                    for (j = i; j < skip * block; j += skip) *ptr++ = data[k * skip * block + j];

                memcpy(data + k * skip * block, data2, skip * block);
            }

            for (ptr = data2, i = 0; i < skip; i++)
                for (j = i; j < bytes - k * skip * block; j += skip) *ptr++ = data[k * skip * block + j];

            memcpy(data + k * skip * block, data2, bytes - k * skip * block);
        }
        else
        {
            for (k = 0; k < bytes / skip / block; k++)
            {
                for (ptr = data + k * skip * block, i = 0; i < skip; i++)
                    for (j = i; j < skip * block; j += skip) data2[j] = *ptr++;

                memcpy(data + k * skip * block, data2, skip * block);
            }

            for (ptr = data + k * skip * block, i = 0; i < skip; i++)
                for (j = i; j < bytes - k * skip * block; j += skip) data2[j] = *ptr++;

            memcpy(data + k * skip * block, data2, bytes - k * skip * block);
        }
    }

    free(data2);
}

void DDS_interleave(uchar* data, uint bytes, uint skip, uint block = 0)
{
    DDS_deinterleave(data, bytes, skip, block, true);
}

void DDS_decode(uchar* chunk, uint size,
    uchar** data, uint* bytes,
    uint block)
{
    uint skip, strip;

    uchar* ptr1, * ptr2;

    uint cnt, cnt1, cnt2;
    int bits, act;

    DDS_initbuffer();

    DDS_clearbits();
    DDS_loadbits(chunk, size);

    skip = DDS_readbits(2) + 1;
    strip = DDS_readbits(16) + 1;

    ptr1 = ptr2 = NULL;
    cnt = act = 0;

    while ((cnt1 = DDS_readbits(DDS_RL)) != 0)
    {
        bits = DDS_readbits(3);
        bits = bits >= 1 ? bits + 1 : bits;

        for (cnt2 = 0; cnt2 < cnt1; cnt2++)
        {
            if (strip == 1 || cnt <= strip) act += DDS_readbits(bits) - (1 << bits) / 2;
            else act += *(ptr2 - strip) - *(ptr2 - strip - 1) + DDS_readbits(bits) - (1 << bits) / 2;

            while (act < 0) act += 256;
            while (act > 255) act -= 256;

            if ((cnt & (DDS_BLOCKSIZE - 1)) == 0)
                if (ptr1 == NULL)
                {
                    if ((ptr1 = (uchar*)malloc(DDS_BLOCKSIZE)) == NULL) 
                        exit(EXIT_FAILURE);
                    ptr2 = ptr1;
                }
                else
                {
                    if ((ptr1 = (uchar*)realloc(ptr1, cnt + DDS_BLOCKSIZE)) == NULL) 
                        exit(EXIT_FAILURE);
                    ptr2 = &ptr1[cnt];
                }

            *ptr2++ = act;
            cnt++;
        }
    }

    if (ptr1 != NULL && (ptr1 = (uchar*)realloc(ptr1, cnt)) == NULL)
        exit(EXIT_FAILURE);

    DDS_interleave(ptr1, cnt, skip, block);

    *data = ptr1;
    *bytes = cnt;
}

uchar* readDDSfile(std::ifstream& file, uint* bytes, uint block)
{
    int cnt;

    uchar* chunk, * data;
    uint size;

    if ((chunk = readRAWfile(file, &size)) == NULL)
        exit(EXIT_FAILURE);

    DDS_decode(chunk, size, &data, bytes, block);

    free(chunk);

    return data;
}

#pragma endregion

uchar* readPVM(const char* path, int& width, int& height, int& depth)
{
    uchar* data;
    char* ptr;
    uint bytes, numc, block;

    bool is_DDS = false;

    int version = 1;

    float sx = 1.0f, sy = 1.0f, sz = 1.0f;

    uint len1 = 0, len2 = 0, len3 = 0, len4 = 0;

    std::ifstream file(path, std::ios::binary);
    if (!file.good())
        exit(EXIT_FAILURE);

    std::string header;
    std::getline(file, header);
    if (strcmp(header.c_str(), DDS_ID) == 0 || strcmp(header.c_str(), DDS_ID2) == 0)
        block = DDS_INTERLEAVE, is_DDS = true;

    if (is_DDS)
        data = readDDSfile(file, &bytes, block);
    else
        data = readRAWfile(file, &bytes);

    if (bytes < 5)
        return nullptr;

    ptr = strchr((char*)data, '\n') + 1;;
    if (!is_DDS)
    {
        if (strcmp(header.c_str(), "PVM2") == 0) version = 2;
        else if (strcmp((char*)data, "PVM3") == 0) version = 3;
        else return(NULL);

        if (sscanf_s((char*)ptr, "%d %d %d\n%g %g %g\n", &width, &height, &depth, &sx, &sy, &sz) != 6)
            exit(EXIT_FAILURE);
        if (width < 1 || height < 1 || depth < 1 || sx <= 0.0f || sy <= 0.0f || sz <= 0.0f)
            exit(EXIT_FAILURE);
        ptr = strchr((char*)ptr, '\n') + 1;
    }
    else
    {
        if (sscanf_s((char*)ptr, "%d %d %d\n", &width, &height, &depth) != 3)
            exit(EXIT_FAILURE);
        if (width < 1 || height < 1 || depth < 1)
            exit(EXIT_FAILURE);
        ptr = strchr((char*)ptr, '\n') + 1;
    }

    ptr = strchr((char*)ptr, '\n') + 1;
    if (sscanf_s((char*)ptr, "%d\n", &numc) != 1) exit(EXIT_FAILURE);
    if (numc < 1) exit(EXIT_FAILURE);

    if (numc != 1)
        exit(EXIT_FAILURE);

    ptr = strchr((char*)ptr, '\n') + 1;
    if (version == 3)
    {
        len1 = strlen((char*)(ptr + (width) * (height) * (depth)*numc)) + 1;
        len2 = strlen((char*)(ptr + (width) * (height) * (depth)*numc + len1)) + 1;
        len3 = strlen((char*)(ptr + (width) * (height) * (depth)*numc + len1 + len2)) + 1;
        len4 = strlen((char*)(ptr + (width) * (height) * (depth)*numc + len1 + len2 + len3)) + 1;
    }

    uchar* volume = new uchar[width * height * depth * numc + len1 + len2 + len3 + len4];
    //if (data + bytes != (uchar*)ptr + (width) * (height) * (depth)*numc + len1 + len2 + len3 + len4)
    //    exit(EXIT_FAILURE);

    memcpy(volume, ptr, (width) * (height) * (depth)*numc + len1 + len2 + len3 + len4);
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

uchar* readVolume(const char* path, int& width, int& height, int& depth)
{
    switch (getFileFormat(path))
    {
    case Format::NRRD:
        return readNRRD(path, width, height, depth);
        break;
    case Format::PVM:
        return readPVM(path, width, height, depth);
        break;
    default:
        break;
    }
}