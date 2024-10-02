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
#define MARGEN 200

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


// Función para recortar la imagen alrededor del tronco
cv::Mat recortar_tronco(cv::Mat& img) 
{
	double centerX;

	   // Convertir a escala de grises
	cv::Mat gray;
	//cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	gray = img;


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
	    cv::Mat column = gray.col(x);
	    cv::Scalar mean, stddev;
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
        //return 0;
        centerX = img.cols / 2;
    } else {


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
	    if (centerX == 0) {
	        cout << "No se encontró un tronco claro" << endl;
	        //return 0;
	        centerX = img.cols / 2;
	    }
	}

    // **NUEVA PARTE**: Mejorar detección de yuyos con segmentación de color
    /*
    int yStart = img.rows * 0.8;  // Analizar el último 20% de la imagen
    cv::Mat bottomRegion = img(cv::Range(yStart, img.rows), cv::Range(bestRegionStart, bestRegionEnd));

	cout << " VA POR 14 "<< endl << flush ;
    // Convertir a espacio de color HSV
    cv::Mat hsv;
    cvtColor(bottomRegion, hsv, cv::COLOR_BGR2HSV);

    // Rango para color verde de los yuyos
    cv::Scalar lowerGreen(35, 40, 40);  // Umbral inferior
    cv::Scalar upperGreen(85, 255, 255);  // Umbral superior

	cout << " VA POR 14 "<< endl << flush ;
    cv::Mat mask;
    inRange(hsv, lowerGreen, upperGreen, mask);

	cout << " VA POR 14 "<< endl << flush ;
    // Calcular la proporción de píxeles verdes
    double greenRatio = (double)countNonZero(mask) / (mask.rows * mask.cols);

    if (greenRatio > 0.2) {  // Ajustar umbral según resultados
        cout << "Yuyos detectados debajo del tronco, posible falso positivo." << endl;
        //return 0;
        centerX = img.cols / 2;
    }
	cout << " VA POR 14 "<< endl << flush ;



	*/






    // Ajustar los límites de recorte para mantener el punto rojo en el centro
    int xLeft = max(0, (int)(centerX - MARGEN));
    int xRight = min(img.cols, (int)(centerX + MARGEN));

    // Ajustar los límites si la región de recorte se sale del borde de la imagen
    if (xLeft == 0) {
        xRight = min(MARGEN*2, img.cols);
    } else if (xRight == img.cols) {
        xLeft = max(0, img.cols - MARGEN*2);
    }

    // Recortar la imagen
    cv::Rect roi(xLeft, 0, xRight - xLeft, img.rows);
    cv::Mat croppedImg = img(roi);

    // Dibujar un círculo rojo en el centro del recorte (opcional)
    //circle(croppedImg, cv::Point(100, croppedImg.rows / 2), 5, cv::Scalar(0, 0, 255), -1);

    // Devolver la imagen recortada
    return croppedImg;
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

void convertir (const char * origen)
{
        char comando[256];
	// sprintf(comando, "convert %s -gravity Center -crop 60%cx100%c+0+0 +repage /tmp/output2.png", origen, '%', '%');
	// system(comando);

	// sprintf(comando, "convert %s   -gravity North -region 100%cx50%c -fill black -colorize 100%c /tmp/output.png", origen, '%', '%', '%');
	// system(comando);

	sprintf(comando, "convert %s -gravity South -crop 100%cx50%c+0+0 +repage /tmp/m/output.png  ", origen, '%', '%');
	system(comando);

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

	// montamos el ram fs
	// system("mkdir /tmp/m");
	// printf("clave de root:\n"); fflush(0);
	// system("sudo mount -t tmpfs ramfs /tmp/m ");

	cout << "Extracting ORB features..." << endl;

	for(int i = 0; i < NIMAGES; ++i) {
		stringstream ss;
		ss << "f" << i << ".jpg";

		//cv::Mat image = cv::imread(ss.str(), 0);
//		cv::Mat image = cv::imread(ss.str(), cv::IMREAD_COLOR);

		// si hay que convertir
		// std::string str = ss.str();
		// convertir(str.c_str());
		// cv::Mat image = cv::imread("/tmp/m/output.png", cv::IMREAD_COLOR);
    // Cargar la imagen en escala de grises
    cv::Mat image = cv::imread(ss.str(), cv::IMREAD_GRAYSCALE);
    //cv::Mat image;
    //cv::Mat image2 = cv::imread(ss.str(), cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "No se pudo cargar la imagen." << std::endl;
        exit(1);
    }

    image = recortar_tronco(image);
    //Mat recortada image = recortar_tronco(image);
		//cv::imshow("ORB Keypoints", image);
		//cv::waitKey(0);


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


		// Dibujar los puntos clave en la imagen
		// cv::Mat image_with_keypoints;
		// drawKeypoints(image, keypoints, image_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

		// Mostrar la imagen con los puntos clave detectados
		// cv::namedWindow("ORB Keypoints", cv::WINDOW_NORMAL);
		// cv::imshow("ORB Keypoints", image_with_keypoints);
		// cv::waitKey(0);


		features.push_back(vector<cv::Mat >());
		changeStructure(descriptors, features.back());
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

