//
//  stereo_pgm_to_png
//  FerProjekt
//
//  Usage: ./stereo_pgm_to_png source_image dest_image_prefix
//  Note: _(left/right).png suffix of dest file will be added automaticaly
//

#include <iostream>
#include <string>
//#include <ifstream>
#include <bitset>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <png.h>

#define IMG_WIDTH    640
#define IMG_HEIGHT   480
#define IMG_SIZE     IMG_WIDTH*IMG_HEIGHT

using namespace std;

/* Loads PGRFly stereo P5 PGM */
void load_stereo_raw(const char* path, uint8_t*** left, uint8_t*** right);

/* Saves raw data to greyscale png document */
void save_to_png(const char * path, uint8_t ** data);

/* http://www.ptgrey.com/support/kb/data/PGRFlyCaptureTimestampTest.cpp */
inline double imageTimeStampToSeconds(unsigned int uiRawTimestamp)
{
    int nSecond      = (uiRawTimestamp >> 25) & 0x7F;   // get rid of cycle_* - keep 7 bits
    int nCycleCount  = (uiRawTimestamp >> 12) & 0x1FFF; // get rid of offset
    int nCycleOffset = (uiRawTimestamp >>  0) & 0xFFF;  // get rid of *_count
    
    //cout << nSecond << endl << nCycleCount << endl << nCycleOffset << endl;
    
    return (double)nSecond + (((double)nCycleCount+((double)nCycleOffset/3072.0))/8000.0);
}


uint32_t read_timestamp(uint8_t **left, uint8_t **right)
{
   uint32_t stamp;
   unsigned char *pStamp = (unsigned char*) &stamp;

   // Little-Endian order
   pStamp[3] = left[0][0];
   pStamp[2] = right[0][0];
   pStamp[1] = left[0][1];
   pStamp[0] = right[0][1];

   uint32_t second_count;
   uint32_t cycle_count;
   uint32_t cycle_offset;

   second_count = (stamp >> 25) & 0x7F; // get rid of cycle_* - keep 7 bits
   cycle_count = (stamp >> 12) & 0x1FFF; // get rid of offset - keep 13 bits
   cycle_offset = (stamp >> 0) & 0xFFF; // get rid of *_count - keep 12 bits

   bitset<7> sec_bits(second_count);
   bitset<13> cycle_bits(cycle_count);
   bitset<12> offset_bits(cycle_offset);   
   cout << sec_bits.to_string() << " -- ";
   cout << cycle_bits.to_string() << " -- ";
   cout << offset_bits.to_string() << endl;

   printf("%u\t%u\t%u\t%08X\n", second_count, cycle_count, 
         cycle_offset, stamp);
   return stamp;
}


int main(int argc, const char * argv[])
{
    uint8_t **left, **right;
    uint32_t timestamp;
    
    if (argc != 3) {
        std::cout << "Invalid number of arguments." << std::endl;
        std::cout << "Usage: " << argv[0] << " source_image dest_image_prefix" << std::endl;
        std::cout << "Will output dest_image_prefix_(left/right).png images." << std::endl;
        return -1;
    }
    
    load_stereo_raw(argv[1], &left, &right);
    
    save_to_png((std::string(argv[2]) + "_left.png").c_str(), left);
    save_to_png((std::string(argv[2]) + "_right.png").c_str(), right);
    
    // TODO: Do something with timestamps
    timestamp = read_timestamp(left, right);
    std::cout << imageTimeStampToSeconds(timestamp) << " s\n";
    
    // Free memory
    for (unsigned row = 0; row < IMG_HEIGHT; row++) {
        delete[] left[row];
        delete[] right[row];
    }
    delete[] left;
    delete[] right;
    
    return 0;
}



/* Loads PGRFly stereo P5 PGM */
void load_stereo_raw(const char *path, uint8_t ***left, uint8_t ***right)
{
    //char simage[IMG_SIZE];
    // Open file
    FILE* file = fopen(path, "rb");

    // Allocate buffers */
    *left = new uint8_t* [IMG_HEIGHT];
    *right = new uint8_t* [IMG_HEIGHT];
    
    // Read image rows
    for (unsigned row = 0; row < IMG_HEIGHT; row++) {
        (*left)[row] = new uint8_t [IMG_WIDTH];
        (*right)[row] = new uint8_t [IMG_WIDTH];
        
        for (unsigned col = 0; col < IMG_WIDTH; col++) {
            fread(&(*left)[row][col], 1, 1, file);
            fread(&(*right)[row][col], 1, 1, file);
        }
    }
    
    fclose(file);
}

/* Saves raw data to greyscale png document */
void save_to_png(const char *path, uint8_t **data)
{
    png_structp png_ptr;
    png_infop info_ptr;
    uint32_t height = IMG_HEIGHT;
    uint32_t width = IMG_WIDTH;
    
    // Open file
    FILE *fp = fopen(path, "wb");
    assert(fp);
    
    // Initialize
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    
    info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    
    setjmp(png_jmpbuf(png_ptr));
    png_init_io(png_ptr, fp);
    
    // Write header
    setjmp(png_jmpbuf(png_ptr));
    
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_GRAY,   // Grayscale image
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE,
                 PNG_FILTER_TYPE_BASE);
    
    png_write_info(png_ptr, info_ptr);
    
    // Write bytes
    setjmp(png_jmpbuf(png_ptr));
    png_write_image(png_ptr, (png_bytep *)data);
    
    // End write
    setjmp(png_jmpbuf(png_ptr));
    png_write_end(png_ptr, NULL);
        
    fclose(fp);
}
