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
#include <deque>

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
const int NIMAGES = 79;
#define MARGEN 300

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

double centerXX = 0.0;



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
		centerXX = centerX;
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

void loadFeatures(vector<vector<cv::Mat > > &features);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void saveBowVectors(const vector<vector<cv::Mat > > &features, const std::string& filename);




void adjustImageToMean(cv::Mat& image, double target_mean) {
    // Calcular el promedio actual de la imagen
    cv::Scalar current_mean_scalar = cv::mean(image);
    double current_mean = current_mean_scalar[0];

    // Calcular la diferencia entre el promedio deseado y el actual
    double shift = target_mean - current_mean;

    // Ajustar la imagen sumando la diferencia
    image.convertTo(image, -1, 1, shift);

    // Recortar los valores para que estén en el rango 0-255
    cv::threshold(image, image, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(image, image, 0, 0, cv::THRESH_TOZERO);
}



// Función para aplicar la Transformada de Retinex multiescala
void applyMSRCR(const cv::Mat& input, cv::Mat& output) {
    cv::Mat logImage;
    cv::Mat retinexImage = cv::Mat::zeros(input.size(), CV_32F);

    // Convertir la imagen a logaritmo para simular la percepción humana de la luz
    cv::Mat floatImage;
    input.convertTo(floatImage, CV_32F, 1.0 / 255.0);  // Convertir a flotante y normalizar
    floatImage += 1.0;  // Evitar logaritmo de cero
    cv::log(floatImage, logImage);

    // Usar filtros gaussianos de diferentes tamaños para realizar Retinex multiescala
    std::vector<cv::Mat> scales(3);
    cv::GaussianBlur(logImage, scales[0], cv::Size(7, 7), 30);
    cv::GaussianBlur(logImage, scales[1], cv::Size(21, 21), 150);
    cv::GaussianBlur(logImage, scales[2], cv::Size(31, 31), 300);

    // Promediar las escalas de Retinex
    for (size_t i = 0; i < scales.size(); ++i) {
        retinexImage += (logImage - scales[i]) / scales.size();
    }

    // Convertir de vuelta a espacio de valores originales
    cv::exp(retinexImage, retinexImage);
    retinexImage -= 1.0;

    // Normalizar el rango dinámico de la imagen resultante
    cv::normalize(retinexImage, retinexImage, 0, 255, cv::NORM_MINMAX);

    retinexImage.convertTo(output, CV_8U);  // Convertir la imagen de nuevo a 8 bits
}






// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}


// ----------------------------------------------------------------------------

int main()
{
  vector<vector<cv::Mat > > features;
  loadFeatures(features);

  saveBowVectors(features, "db_bowvectors1.dat");

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<cv::Mat > > &features)
{
	features.clear();
	features.reserve(NIMAGES);

  //cv::Ptr<cv::ORB> orb = cv::ORB::create(8000);
  // cv::Ptr<cv::ORB> orb = cv::ORB::create(200, 1.01, 9, 55, 2, 2, cv::ORB::HARRIS_SCORE, 85);
  // BIEN cv::Ptr<cv::ORB> orb = cv::ORB::create(200, 1.01, 5, 90, 1, 2, cv::ORB::HARRIS_SCORE, 30);
  // EL MEJOR! BUEN PERFORMANCE cv::Ptr<cv::ORB> orb = cv::ORB::create(200, 1.01, 3, 65, 2, 4, cv::ORB::HARRIS_SCORE, 45);
  // ULTIMO MUY BIEN cv::Ptr<cv::ORB> orb = cv::ORB::create(200, 1.01, 15, 85, 2, 4, cv::ORB::HARRIS_SCORE, 75);
  //
  //cv::Ptr<cv::ORB> orb = cv::ORB::create(300, 1.01, 3, 65, 2, 4, cv::ORB::HARRIS_SCORE, 45);
  //cv::Ptr<cv::ORB> orb = cv::ORB::create(400, 1.01, 15, 85, 2, 4, cv::ORB::HARRIS_SCORE, 75);
  cv::Ptr<cv::ORB> orb = cv::ORB::create(400, 1.01, 3, 65, 2, 4, cv::ORB::HARRIS_SCORE, 45);



    std::deque<std::string> ultimas_lineas;
    const int n_lineas = 5;


  // buscar troncos
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

	int n_arbol = 0;
	int ya_esta = 0;
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

        // Guarda las últimas 4 líneas en una cola (deque)
        if (ultimas_lineas.size() == n_lineas) {
            ultimas_lineas.pop_front();  // Elimina la más antigua
        }
        ultimas_lineas.push_back(linea);


		cv::Mat image = cv::imread(ss.str(), cv::IMREAD_GRAYSCALE);
		if (image.empty()) {
			std::cerr << "No se pudo cargar la imagen." << std::endl;
			exit(1);
		}

		if (!recortar_tronco(image, image)) {
			total = 0;
			ya_esta = 0;
			continue;
		}
		total++;
		if ((total >= 10) && (centerXX > 500) && (centerXX < 600) && (ya_esta == 0)) {
                	std::cout << " :tronco detectado. " << total << " " << ss.str();

			ya_esta = 1;

			for (const auto& l : ultimas_lineas) {

				n_arbol++;
                	std::cout << " arbol-db: " << n_arbol << " " << l << " " << ss.str() << std::endl;


    // Cargar la imagen en escala de grises
    image = cv::imread(l, cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "No se pudo cargar la imagen." << std::endl;
        exit(1);
    }

    recortar_tronco(image, image);

    // Aplicar la Transformada de Retinex multiescala
    cv::Mat retinexImage;
    applyMSRCR(image, retinexImage);

    // Ajustar el brillo para mejorar la visibilidad
    cv::Mat finalImage;
    retinexImage.convertTo(finalImage, -1, 1.5, 50);  // Incrementar contraste y brillo

    double target_mean = 128.0;

    // Ajustar las imágenes para que tengan el promedio deseado
    adjustImageToMean(finalImage, target_mean);

    // Reemplazar la imagen original con la imagen final
    image = finalImage.clone();  // Hacer una copia de la imagen final en la imagen original

		cv::Mat mask;
		vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;

		orb->detectAndCompute(image, mask, keypoints, descriptors);


		features.push_back(vector<cv::Mat >());
		changeStructure(descriptors, features.back());
	}
	}
	}
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------



void saveBowVectors(const vector<vector<cv::Mat > > &features, const std::string& filename) {
	// branching factor and depth levels 
	//const int k = 9;
	//const int L = 3;
	// const int k = 15;
	const int k = 50;
	const int L = 3;
	const WeightingType weight = TF_IDF;
	const ScoringType scoring = L1_NORM;

	OrbVocabulary voc(k, L, weight, scoring);

	cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
	voc.create(features);
	cout << "... done!" << endl;

	cout << "Vocabulary information: " << endl
	<< voc << endl << endl;

	std::vector<BowVector> bowVectors(NIMAGES);

	// Precomputar los BowVector
	for (int i = 0; i < NIMAGES; i++) {
    		voc.transform(features[i], bowVectors[i]);
	}

	// Guardar vocabulario
	//voc.save("db_vocabulary1.yml.gz");  // Guardar en un archivo comprimido
	voc.save("db_vocabulary1.yml");  // Guardar en un archivo comprimido


	// Guardar los BowVectors en un archivo
	//saveBowVectors(bowVectors, "bowvectors1.dat");
	
	std::ofstream ofs(filename, std::ios::binary);
	for (const auto& bv : bowVectors) {
		size_t size = bv.size();
		ofs.write(reinterpret_cast<const char*>(&size), sizeof(size)); // Escribir el tamaño del BowVector
		for (const auto& pair : bv) {
			ofs.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));  // Escribir WordId
			ofs.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second)); // Escribir WordValue
        	}
	}
	ofs.close();
}


std::vector<BowVector> loadBowVectors(const std::string& filename) {
	std::ifstream ifs(filename, std::ios::binary);
	std::vector<BowVector> bowVectors;

	while (ifs.peek() != EOF) {
		BowVector bv;
		size_t size;
		ifs.read(reinterpret_cast<char*>(&size), sizeof(size)); // Leer el tamaño del BowVector
		for (size_t i = 0; i < size; ++i) {
			WordId id;
			WordValue value;
			ifs.read(reinterpret_cast<char*>(&id), sizeof(id));     // Leer WordId
			ifs.read(reinterpret_cast<char*>(&value), sizeof(value)); // Leer WordValue
			bv[id] = value;
		}
		bowVectors.push_back(bv);
	}
	ifs.close();
	return bowVectors;
}


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

