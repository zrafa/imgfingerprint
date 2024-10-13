/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <opencv2/ximgproc.hpp>



#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 1262;
#define MARGEN 150


// ------------------------------------------------------------------------------------------------


// Función para recortar la imagen alrededor del tronco
bool recortar_tronco(const cv::Mat& img, cv::Mat& recortada) 
//std::optional<cv::Mat> recortar_tronco(cv::Mat& img) 
{
	double centerX;

	// Convertir a escala de grises
	cv::Mat gray;
	cv::Rect roi2(0, 0, img.cols, img.rows-50);
	gray = img(roi2);
	//gray = img;

	// Aplicar un filtro Gaussiano para reducir el ruido
	cv::Mat blurred;
	GaussianBlur(gray, blurred, cv::Size(5, 5), 2);

	// Detectar bordes usando Canny
	cv::Mat edges;
	Canny(blurred, edges, 20, 60);

	// Aplicar la Transformada de Hough para detectar líneas
	vector<cv::Vec4i> lines;
	HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 50, 5);

	// Filtrar líneas verticales
	vector<cv::Vec4i> verticalLines;
	double dx;
	double dy;
	for (auto& l : lines) {
		dx = abs(l[2] - l[0]);
		dy = abs(l[3] - l[1]);
		double angle = atan2(dy, dx) * 180 / CV_PI;
		//if (angle > 75 && angle < 105 && dy > 50) {
		if (angle > 65 && angle < 115 && dy > 50) {
			verticalLines.push_back(l);
		}
	}


	// Analizar continuidad de color en columnas
	vector<int> lowVarianceColumns;
	for (int x = 0; x < gray.cols; ++x) {
		cv::Mat column = gray.col(x);
		cv::Scalar mean, stddev;
		meanStdDev(column, mean, stddev);

		// Si la desviación estándar es baja, hay poca variación vertic
		// RAFA if (stddev[0] < 20) {
		// RAFA MUY BUENO if (stddev[0] < 10) {
		if (stddev[0] < 15) {
			lowVarianceColumns.push_back(x);
		}
	}

	// Agrupar líneas en regiones densas
	sort(lowVarianceColumns.begin(), lowVarianceColumns.end());

	vector<pair<int, int>> regions;
	if (lowVarianceColumns.empty()) {
		cout << "No se encontró un tronco claro color" << endl;
		return false; // Indicar error
	};
      // 	else {

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
		centerX = (bestRegionStart + bestRegionEnd) / 2.0;
		// RAFA para nuevas pruebas if (centerX == 0) {
		if ((centerX == 0) || (centerX < 250) || (centerX > 614) ) {
			cout << "No se encontró un tronco claro" << endl;
			return false;
		}
		cout << " centerx " << centerX << flush ;
//	}

	// Ajustar los límites de recorte para mantener 
	// el punto rojo en el centro
	int xLeft = max(0, (int)(centerX - MARGEN));
	int xRight = min(img.cols, (int)(centerX + MARGEN));

	// Ajustar los límites si la región de recorte 
	// se sale del borde de la imagen
	if (xLeft == 0) {
		xRight = min(MARGEN*2, img.cols);
	} else if (xRight == img.cols) {
		xLeft = max(0, img.cols - MARGEN*2);
	}

	// Recortar la imagen
	cv::Rect roi(xLeft, 0, xRight - xLeft, img.rows);
	recortada = img(roi);

	// Dibujar un círculo rojo en el centro del recorte (opcional)
	// circle(recortada, cv::Point(100, recortada.rows / 2), 5, cv::Scalar(0, 0, 255), -1);
	return true;
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void buscar_troncos();






int main()
{
  buscar_troncos();

  return 0;
}

// ----------------------------------------------------------------------------

void buscar_troncos()
{
	std::ifstream archivo("listado.txt");
	if (!archivo.is_open()) {
		std::cerr << "Error al abrir el archivo." << std::endl;
		 exit (1);
	}

	// Leer el número de fotos 
	int numero;
	archivo >> numero;

	// Ignorar el resto de la primera línea (por si hay más datos)
	archivo.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	// vemos si podemos encontrar el arbol

	int total = 0;
  	int i;
	std::chrono::time_point<std::chrono::high_resolution_clock> start;
	for (i=0; i<numero; i++) {

                // Captura el tiempo final
                auto end = std::chrono::high_resolution_clock::now();
                // Calcula la duración
                std::chrono::duration<double> duration = end - start;
                // Imprime la duración en segundos
                std::cout << " Tiempo transcurrido foto : " << i-1 << "  " << duration.count() << " segundos" << std::endl;
                // Inicia un nuevo cronometro
                start = std::chrono::high_resolution_clock::now();


		// Leer las líneas restantes y procesarlas
		std::string linea;
		std::getline(archivo, linea);
		std::stringstream ss(linea);

		cv::Mat image = cv::imread(ss.str(), cv::IMREAD_GRAYSCALE);

		if (image.empty()) {
			std::cerr << "No se pudo cargar la imagen." << std::endl;
			exit(1);
		}

		if (!recortar_tronco(image, image)) {
			total = 0;
			continue;
		}
		total++;
		if (total >= 3)
                	std::cout << " :tronco detectado. " << ss.str(); 
		// cv::imshow("ORB Keypoints", image);
		// cv::waitKey(0);

	}
}

// ----------------------------------------------------------------------------

