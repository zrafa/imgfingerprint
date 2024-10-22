
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

int main() {
    // Leer la imagen en escala de grises
    cv::Mat gray = cv::imread("tronco.jpg", cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
        std::cout << "No se pudo abrir la imagen." << std::endl;
        return -1;
    }

    // Obtener el tamaño de la imagen
    int rows = gray.rows;
    int cols = gray.cols;

    // Columna central
    int centralCol = cols / 2;

    // Calcular la media de grises de la columna central
    double centralMean = 0.0;
    for (int i = 0; i < rows; i++) {
        centralMean += gray.at<uchar>(i, centralCol);
    }
    centralMean /= rows;

    std::cout << "Media de la columna central: " << centralMean << std::endl;

    // Función para calcular la media de grises de una columna
    auto calcularMediaColumna = [&](int col) {
        double mean = 0.0;
        for (int i = 0; i < rows; i++) {
            mean += gray.at<uchar>(i, col);
        }
        return mean / rows;
    };

    // Umbral de diferencia para considerar que los colores no coinciden
    double umbral = 10.0;  // Ajustar este valor según la imagen

    // Buscar los bordes hacia la izquierda y derecha
    int bordeIzquierdo = -1;
    int bordeDerecho = -1;

    // Buscar hacia la izquierda
    for (int col = centralCol - 1; col >= 0; col--) {
        double mediaColumna = calcularMediaColumna(col);
        double mediaColumnaSiguiente = calcularMediaColumna(col - 1);

        if (std::abs(mediaColumna - centralMean) > umbral && std::abs(mediaColumnaSiguiente - centralMean) > umbral) {
            bordeIzquierdo = col;
            break;
        }
    }

    // Buscar hacia la derecha
    for (int col = centralCol + 1; col < cols; col++) {
        double mediaColumna = calcularMediaColumna(col);
        double mediaColumnaSiguiente = calcularMediaColumna(col + 1);

        if (std::abs(mediaColumna - centralMean) > umbral && std::abs(mediaColumnaSiguiente - centralMean) > umbral) {
            bordeDerecho = col;
            break;
        }
    }

    // Mostrar los resultados
    if (bordeIzquierdo != -1 && bordeDerecho != -1) {
        std::cout << "Borde izquierdo detectado en x: " << bordeIzquierdo << std::endl;
        std::cout << "Borde derecho detectado en x: " << bordeDerecho << std::endl;

	        // Dibujar las líneas de los bordes en la imagen
        cv::Mat result;
        cv::cvtColor(gray, result, cv::COLOR_GRAY2BGR);  // Convertir a BGR para dibujar en color
        cv::line(result, cv::Point(bordeIzquierdo, 0), cv::Point(bordeIzquierdo, rows), cv::Scalar(0, 0, 255), 2);  // Línea roja para el borde izquierdo
        cv::line(result, cv::Point(bordeDerecho, 0), cv::Point(bordeDerecho, rows), cv::Scalar(0, 255, 0), 2);  // Línea verde para el borde derecho

        // Mostrar la imagen con los bordes detectados
        cv::imshow("Bordes del tronco detectados", result);
        cv::waitKey(0);
    } else {
        std::cout << "No se detectaron los bordes del tronco." << std::endl;
    }

    return 0;
}

