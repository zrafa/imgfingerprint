
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

    imshow("Resultado", img);
    cv::waitKey(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

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
        if (angle > 75 && angle < 105 && dy > 50) {
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
        // if (stddev[0] < 15) {  // Ajustar umbral según el resultado
        if (stddev[0] < 20) {  // Ajustar umbral según el resultado
            lowVarianceColumns.push_back(x);
        }
    }

    // Agrupar líneas en regiones densas
    sort(lowVarianceColumns.begin(), lowVarianceColumns.end());

    vector<pair<int, int>> regions;
    if (lowVarianceColumns.empty()) { 
	    cout << "no hay tronco" << endl;
	    return 0;
    }

    int start = lowVarianceColumns[0];
    int end = start;
    for (size_t i = 1; i < lowVarianceColumns.size(); ++i) {
        if (lowVarianceColumns[i] - end <= 5) {  // Si las columnas están cerca
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
	    cout << "no hay tronco" << endl;
	    return 0;
    }

    // Dibujar líneas y centro en la imagen
    for (int x = bestRegionStart; x <= bestRegionEnd; ++x) {
        line(img, Point(x, 0), Point(x, img.rows - 1), Scalar(255, 0, 0), 1);
    }
    circle(img, Point(centerX, img.rows / 2), 5, Scalar(0, 0, 255), -1);
    cout << "Coordenada X aproximada del centro del tronco: " << centerX << endl;

    imshow("Resultado", img);
    waitKey(0);

    return 0;
}

