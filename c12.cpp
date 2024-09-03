
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Uso: " << argv[0] << " <nombre_del_archivo>" << endl;
        return -1;
    }

    // Cargar la imagen desde el archivo pasado como argumento
    Mat img = imread(argv[1]);
    if (img.empty()) {
        cerr << "Error al cargar la imagen." << endl;
        return -1;
    }

    // Convertir a escala de grises
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Aplicar un filtro Gaussiano para reducir el ruido
    Mat blurred;
    GaussianBlur(gray, blurred, Size(5, 5), 2);

    // Detectar bordes usando Canny
    Mat edges;
    Canny(blurred, edges, 20, 60);

    // Aplicar la Transformada de Hough para detectar líneas
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 50, 5);

    // Filtrar líneas verticales
    vector<Vec4i> verticalLines;
    for (auto& l : lines) {
        double dx = abs(l[2] - l[0]);
        double dy = abs(l[3] - l[1]);
        double angle = atan2(dy, dx) * 180 / CV_PI;
        //if (angle > 75 && angle < 105 && dy > 50) {
        if (angle > 65 && angle < 115 && dy > 50) {
            verticalLines.push_back(l);
        }
    }

    // Analizar continuidad de color en columnas
    vector<int> lowVarianceColumns;
    for (int x = 0; x < gray.cols; ++x) {
        Mat column = gray.col(x);
        Scalar mean, stddev;
        meanStdDev(column, mean, stddev);

        // Si la desviación estándar es baja, hay poca variación vertical
        if (stddev[0] < 20) {
            lowVarianceColumns.push_back(x);
        }
    }

    // Agrupar líneas en regiones densas
    sort(lowVarianceColumns.begin(), lowVarianceColumns.end());

    vector<pair<int, int>> regions;
    if (lowVarianceColumns.empty()) {
        cout << "No se encontró un tronco claro" << endl;
        return 0;
    }

    int start = lowVarianceColumns[0];
    int end = start;
    for (size_t i = 1; i < lowVarianceColumns.size(); ++i) {
        if (lowVarianceColumns[i] - end <= 5) {
            end = lowVarianceColumns[i];
        } else {
            regions.push_back(make_pair(start, end));
            start = lowVarianceColumns[i];
            end = start;
        }
    }
    regions.push_back(make_pair(start, end));

    // Encontrar la región con mayor densidad de líneas
    int maxDensity = 0;
    int bestRegionStart = 0;
    int bestRegionEnd = 0;
    for (auto& region : regions) {
        int density = region.second - region.first;
        if (density > maxDensity) {
            maxDensity = density;
            bestRegionStart = region.first;
            bestRegionEnd = region.second;
        }
    }

    // Calcular el centro de la región más densa
    double centerX = (bestRegionStart + bestRegionEnd) / 2.0;
    if (centerX == 0) {
        cout << "No se encontró un tronco claro" << endl;
        return 0;
    }

    // **NUEVA PARTE**: Mejorar detección de yuyos con segmentación de color
    int yStart = img.rows * 0.8;  // Analizar el último 20% de la imagen
    Mat bottomRegion = img(Range(yStart, img.rows), Range(bestRegionStart, bestRegionEnd));

    // Convertir a espacio de color HSV
    Mat hsv;
    cvtColor(bottomRegion, hsv, COLOR_BGR2HSV);

    // Rango para color verde de los yuyos
    Scalar lowerGreen(35, 40, 40);  // Umbral inferior
    Scalar upperGreen(85, 255, 255);  // Umbral superior

    Mat mask;
    inRange(hsv, lowerGreen, upperGreen, mask);

    // Calcular la proporción de píxeles verdes
    double greenRatio = (double)countNonZero(mask) / (mask.rows * mask.cols);

    if (greenRatio > 0.2) {  // Ajustar umbral según resultados
        cout << "Yuyos detectados debajo del tronco, posible falso positivo." << endl;
        return 0;
    }

    // Dibujar líneas y centro en la imagen
    for (int x = bestRegionStart; x <= bestRegionEnd; ++x) {
        line(img, Point(x, 0), Point(x, img.rows - 1), Scalar(255, 0, 0), 1);
    }
    circle(img, Point(centerX, img.rows / 2), 5, Scalar(0, 0, 255), -1);
    cout << "Coordenada X aproximada del centro del tronco: " << centerX << endl;

        // Calcular límites del recorte
    int xLeft = max(0, (int)(centerX - 100));
    int xRight = min(img.cols, (int)(centerX + 100));

    // Recortar la imagen
    Rect roi(xLeft, 0, xRight - xLeft, img.rows);
    Mat croppedImg = img(roi);

    // Mostrar la imagen recortada
    imshow("Imagen Recortada", croppedImg);
    waitKey(0);

//    imshow("Resultado", img);
//    waitKey(0);

    return 0;
}

