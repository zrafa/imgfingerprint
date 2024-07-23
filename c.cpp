
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace cv;

int convertir2 (const char *name)
{

    // Cargar la imagen
    Mat image = imread(name, IMREAD_COLOR);
    if (image.empty()) {
        printf("Could not open or find the image\n");
        return -1;
    }

    // Obtener dimensiones de la imagen
    int width = image.cols;
    int height = image.rows;

    // Definir la región central
    int center_width = width / 3;
    int start_x = (width - center_width) / 2;

    // Crear una máscara negra
    Mat mask = Mat::zeros(height, width, image.type());

    // Copiar la región central de la imagen original a la máscara
    Rect roi(start_x, 0, center_width, height);
    image(roi).copyTo(mask(roi));

    // Guardar la imagen resultante
    imwrite("/tmp/output.png", mask);

    return 0;
}




