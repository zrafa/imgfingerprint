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

#include <sstream>
#include <cmath>
#include <limits>


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
#define MARGEN 200


// ---------------------- PINTAR TRONCO 

// Función para calcular la media de un parche central de 10x10 píxeles
double calcularMediaParcheCentral(const cv::Mat& gray, int centroX, int patchSize) {
    int rows = gray.rows;
    int cols = gray.cols;

    int centralRow = rows / 2;
    // int centralCol = cols / 2;
    int centralCol = centroX;

    double sum = 0.0;
    int count = 0;

    for (int i = -patchSize/2; i < patchSize/2; i++) {
        for (int j = -patchSize/2; j < patchSize/2; j++) {
            sum += gray.at<uchar>(centralRow + i, centralCol + j);
            count++;
        }
    }

    return sum / count;
}

// Función para encontrar los límites izquierdo y derecho de una fila
std::pair<int, int> encontrarLimitesFila(const cv::Mat& gray, int centroX, int row, double mediaCentral, double umbral) {
    int cols = gray.cols;
    // int centralCol = cols / 2;
    int centralCol = centroX;

    // Buscar borde izquierdo
    int limiteIzq = -1;
    for (int col = centralCol; col >= 0; col--) {
        if (std::abs(gray.at<uchar>(row, col) - mediaCentral) > umbral) {
            limiteIzq = col;
            break;
        }
    }

    // Buscar borde derecho
    int limiteDer = -1;
    for (int col = centralCol; col < cols; col++) {
        if (std::abs(gray.at<uchar>(row, col) - mediaCentral) > umbral) {
            limiteDer = col;
            break;
        }
    }

    return {limiteIzq, limiteDer};
}

// Función principal para pintar el tronco
void pintarTronco(cv::Mat& img, int centroX, double umbral) {
    // Convertir a escala de grises si no lo está
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }

    int rows = gray.rows;
    int cols = gray.cols;
    int centralRow = rows / 2;
    //int centralRow = centroX;
    int patchSize = 10;

    // Calcular la media de un parche 10x10 en el centro
    double mediaCentral = calcularMediaParcheCentral(gray, centroX, patchSize);

    // Pintar fila por fila, desde la central hacia arriba y hacia abajo
    for (int rowOffset = 0; rowOffset <= centralRow; rowOffset++) {
        // Fila superior e inferior
        int rowUp = centralRow - rowOffset;
        int rowDown = centralRow + rowOffset;

        // Encontrar límites de la fila superior
        auto [limiteIzqUp, limiteDerUp] = encontrarLimitesFila(gray, centroX, rowUp, mediaCentral, umbral);
        auto [limiteIzqDown, limiteDerDown] = encontrarLimitesFila(gray, centroX, rowDown, mediaCentral, umbral);

        // Pintar filas superior e inferior (en verde)
        if (limiteIzqUp != -1 && limiteDerUp != -1) {
            cv::line(img, cv::Point(limiteIzqUp, rowUp), cv::Point(limiteDerUp, rowUp), cv::Scalar(0, 255, 0), 1);
            img.at<cv::Vec3b>(rowUp, cols / 2) = cv::Vec3b(0, 0, 255);  // Punto central en rojo
        }
        if (limiteIzqDown != -1 && limiteDerDown != -1) {
            cv::line(img, cv::Point(limiteIzqDown, rowDown), cv::Point(limiteDerDown, rowDown), cv::Scalar(0, 255, 0), 1);
            img.at<cv::Vec3b>(rowDown, cols / 2) = cv::Vec3b(0, 0, 255);  // Punto central en rojo
        }
    }
        cv::imshow("Bordes del tronco detectados", img);
        cv::waitKey(0);
}



// ---------------------- ENCONTRAR TRONCO

// Estructura para mientras se identifica un arbol con info util
struct frutal {
	int nro_arbol;
	int distancia;
	int diametro;
	int x1; 	// lateral izquierdo del arbol en la foto
	int x2; 	// lateral derecho del arbol en la foto
	cv::Mat image;  // falta foto
	// falta marca de tiempo
} st_frutal;

#define N_ULT_ARBOLES 5
frutal ultimos_arboles[N_ULT_ARBOLES];

void ult_arboles_init(void )
{
	int i;
	for (i=0; i<N_ULT_ARBOLES; i++){
		ultimos_arboles[i].nro_arbol = -1;
		ultimos_arboles[i].distancia = -1;
		ultimos_arboles[i].diametro = -1;
		ultimos_arboles[i].x1 = -1;
		ultimos_arboles[i].x2 = -1;
	}
}



// Estructura para almacenar la información del lidar
struct LidarData {
    int distancia;
    int tiempo_ms;
    long long marca_ms;
    long long marca_us;
};

    // Leer los datos del archivo lidar.txt
    //std::vector<LidarData> datosLidar = leerDatosLidar("lidar.txt");
std::vector<LidarData> datosLidar;


// Función para leer los datos del archivo lidar.txt
std::vector<LidarData> leerDatosLidar(const std::string& nombreArchivo) {
    std::vector<LidarData> datos;
    std::ifstream archivo(nombreArchivo);
    std::string linea;
    std::string campo1, campo2, campo3;

    while (std::getline(archivo, linea)) {
        std::stringstream ss(linea);
        std::string token;
        LidarData data;

        // Parsear la línea
        ss >> campo1 >> data.marca_us >> data.marca_ms;

        // Extraer la distancia y el tiempo desde el primer campo
        std::stringstream ss_campo1(campo1);
        std::string aux;
        std::getline(ss_campo1, aux, ':');  // 000
        std::getline(ss_campo1, aux, ':');  // 00102 (distancia)
        data.distancia = std::stoi(aux);
        std::getline(ss_campo1, aux, ':');  // 000002 (tiempo de demora)
        data.tiempo_ms = std::stoi(aux);

            // Aplicar la condición de distancia y tiempo
            if (data.distancia < 200 && data.tiempo_ms > 10) {
                data.distancia = 400;
	    }
        datos.push_back(data);
    }
    return datos;
}


/*
long long buscarDistanciaCercana(long long marcaTiempo_us) {
    std::ifstream archivo("lidar.txt");
    std::string linea;
    long long marcaTiempoMasCercana = 0;
    int distanciaMasCercana = 400;  // Valor por defecto si no se encuentra una distancia válida
    long long diferenciaMinima = LLONG_MAX;

    while (std::getline(archivo, linea)) {
        std::stringstream ss(linea);
        std::string campo1, campo2, campo3;
        long long marcaTiempoLidar_us;
        int distancia;
        int tiempoMs;

        // Parsear la línea
        ss >> campo1 >> marcaTiempoLidar_us >> campo3;

        // Extraer la distancia y el tiempo desde el primer campo
        std::stringstream ss_campo1(campo1);
        std::string aux;
        std::getline(ss_campo1, aux, ':');  // 000
        std::getline(ss_campo1, aux, ':');  // 00102 (distancia)
        distancia = std::stoi(aux);
        std::getline(ss_campo1, aux, ':');  // 000002 (tiempo de demora)
        tiempoMs = std::stoi(aux);

        // Calcular la diferencia de tiempo entre la marca de tiempo buscada y la del archivo
        long long diferencia = std::abs(marcaTiempo_us - marcaTiempoLidar_us);

        // Si es la marca de tiempo más cercana hasta ahora
        if (diferencia < diferenciaMinima) {
            diferenciaMinima = diferencia;
            marcaTiempoMasCercana = marcaTiempoLidar_us;

            // Aplicar la condición de distancia y tiempo
            if (distancia < 200 && tiempoMs > 10) {
                distanciaMasCercana = 400;
            } else {
                distanciaMasCercana = distancia;
            }
        }
    }

    archivo.close();
    return distanciaMasCercana;
}
*/


// Función para buscar la distancia más cercana dada una marca de tiempo
//int buscarDistanciaCercana(const std::vector<LidarData>& datos, long long tiempo_us) {
int buscarDistanciaCercana(long long tiempo_us) {
    int distanciaCercana = -1;
    long long menorDiferencia = std::numeric_limits<long long>::max();

    for (const auto& dato : datosLidar) {
        long long diferencia = std::abs(dato.marca_us - tiempo_us);

        if (diferencia < menorDiferencia) {
            menorDiferencia = diferencia;
            // Aplicar la regla de distancia
	    /*
            if (dato.distancia < 200 && dato.tiempo_ms > 10) {
                distanciaCercana = 400;  // Reemplazar por 400 cm
            } else {
                distanciaCercana = dato.distancia;
            }
	    */
                distanciaCercana = dato.distancia;
        }
    }

    return distanciaCercana;
}




// ------------------------------------------------------------------------------------------------

void encontrar_bordes(const cv::Mat& img, long long marca_tiempo, int *x1, int *x2) 
{

        cv::Mat gray = img;

	    // Obtener las dimensiones de la imagen
    int rows = gray.rows;
    int cols = gray.cols;

    // Columna central
    int centralCol = cols / 2;




    // Umbral de diferencia para la homogeneidad de los píxeles
    double umbralHomogeneidad = 15.0; // Ajustar este valor según la imagen

    // Función para calcular la media de gris de una columna
    auto calcularMediaColumna = [&](int col) {
        double suma = 0.0;
        for (int i = 0; i < rows; i++) {
            suma += gray.at<uchar>(i, col);
        }
        return suma / rows;
    };

    // Media de la columna central
    double mediaCentral = calcularMediaColumna(centralCol);

    // Función para verificar si una columna es homogénea comparando con su media
    auto esColumnaHomogenea = [&](int col, double mediaColumna, double umbral) {
        int countSimilar = 0;
        for (int i = 0; i < rows; i++) {
            if (std::abs(gray.at<uchar>(i, col) - mediaColumna) < umbral) {
                countSimilar++;
            }
        }
        std::cout << "columna central :" << countSimilar*100/rows << std::endl;
        // Verificar si al menos el 70% de los píxeles son similares a la media
        return (countSimilar > 0.8 * rows);
    };

    // Verificar si la columna central es homogénea
 //   if (!esColumnaHomogenea(centralCol, mediaCentral, umbralHomogeneidad)) {
  //      std::cout << "La columna central no es homogénea, por lo tanto, no parece ser un tronco." << std::endl;
//	return;
 //   }

    // int distancia = buscarDistanciaCercana(datosLidar, marca_tiempo);
    int distancia = buscarDistanciaCercana(marca_tiempo);

    if (distancia > 200) {
        std::cout << "Distancia lejana: ." << distancia << " MARCA TIEMPO: " << marca_tiempo << std::endl;
	return;
    }

    // Umbral para la diferencia de nivel de gris entre píxeles
    double umbral = 15.0; // Ajustar según el contraste de la imagen

    // Función para calcular la media de gris entre dos píxeles
    auto diferenciaPixel = [&](int fila, int col1, int col2) {
        return std::abs(gray.at<uchar>(fila, col1) - gray.at<uchar>(fila, col2));
    };

    // Función para verificar si más del 50% de los píxeles de la columna difieren de la columna adyacente
    auto columnaEsBorde = [&](int col1, int col2) {
        int countDiff = 0;
        for (int i = 0; i < rows; i++) {
            if (diferenciaPixel(i, col1, col2) > umbral) {
                countDiff++;
            }
        }
        // Verificar si más del 50% de los píxeles son diferentes
        return (countDiff > (50*rows/100));
    };

    int bordeIzquierdo = -1;
    int bordeDerecho = -1;

    // Buscar borde izquierdo desde la columna central hacia la izquierda
    for (int col = centralCol - 1; col > 0; col--) {
        //if (columnaEsBorde(col, col + 1)) {
        if (columnaEsBorde(col, centralCol)) {
            bordeIzquierdo = col;
            break;
        }
    }

    // Buscar borde derecho desde la columna central hacia la derecha
    for (int col = centralCol + 1; col < cols - 1; col++) {
        //if (columnaEsBorde(col, col - 1)) {
        if (columnaEsBorde(col, centralCol)) {
            bordeDerecho = col;
            break;
        }
    }

    // Mostrar los resultados
    if (bordeIzquierdo != -1 && bordeDerecho != -1) {
        std::cout << "Borde izquierdo detectado en x: " << bordeIzquierdo << std::endl;
        std::cout << "Borde derecho detectado en x: " << bordeDerecho << std::endl;
        std::cout << "Distancia:  " << distancia << std::endl;
	*x1 = bordeIzquierdo;
	*x2 = bordeDerecho;

	        // Dibujar las líneas de los bordes en la imagen
        cv::Mat result;
        cv::cvtColor(gray, result, cv::COLOR_GRAY2BGR);  // Convertir a BGR para dibujar en color
        cv::line(result, cv::Point(bordeIzquierdo, 0), cv::Point(bordeIzquierdo, rows), cv::Scalar(0, 0, 255), 2);  // Línea roja para el borde izquierdo
        cv::line(result, cv::Point(bordeDerecho, 0), cv::Point(bordeDerecho, rows), cv::Scalar(0, 255, 0), 2);  // Línea verde para el borde derecho

	//pintarTronco(result, (bordeDerecho-bordeIzquierdo)/2+bordeIzquierdo, 15);
        // Mostrar la imagen con los bordes detectados
        cv::imshow("Bordes del tronco detectados", result);
        cv::waitKey(0);
    } else {
        std::cout << "No se detectaron los bordes del tronco." << std::endl;
    }

}



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

	/*
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
	*/


	// Analizar continuidad de color en columnas
	vector<int> lowVarianceColumns;
	for (int x = 0; x < gray.cols; ++x) {
		cv::Mat column = gray.col(x);
		cv::Scalar mean, stddev;
		meanStdDev(column, mean, stddev);

		// Si la desviación estándar es baja, hay poca variación vertic
		// RAFA if (stddev[0] < 20) {
		// RAFA MUY BUENO if (stddev[0] < 15) {
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
datosLidar = leerDatosLidar("lidar.txt");
  buscar_troncos();

  return 0;
}

// IDEAS:
//   tener un arreglo de 4 arboles con los datos:
//      - nro de arbol en la hilera
//      - distancia
//      - diametro
//      - tal vez orb descriptors
//
//   Entonces, si se detectó un tronco al menos 3 veces con distancia acorde, entonces
//   registrar los 4 siguientes arboles (siempre que cumplan la condiciones:
//   - hay tronco en la foto
//   - la distancia es acorde
//   - tambien registrar el diametro en el arreglo.
//
//   Si los diametros coinciden y son "mas o menos interesantes (no muy delgados)", registrar
//   en la BD:
//        NRO de arbol en la hilera
//        diametro
//        los 4 fingerprints para el mismo arbol
//
//   Cuando se busque un arbol en la BD, solo existiran datos (en la BD) de algunos arboles de la hilera. 
//   Arboles interesantes (los delgados o con diametros que fluctuaron no estarán en la BD)
//
//   Entonces el algoritmo de posicionamiento será así:
//       - por un lado, cuando se detecte un arbol en la foto 3 veces, con distancia acorde,
//         se contará + 1 (luego tiene que venir un periodo de "no distancia", para volver a contar un arbol
//         Lo anterior intentará posicionarse "contando" los arboles en la hilera.
//       - en paralelo, cuando el arbol parezca interesante (diametro parejo, distancia acorde, etc).
//         se intentará buscar ese arbol en la BD (por diametro, orb descriptors).
//
//       
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
	int x1, x2;  // posible borde de un arbol
	int arbol = 0;  // nro de arbol en la hilera
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


    std::string nombreArchivo = ss.str();
    // Encontrar la posición del punto para eliminar la extensión
    size_t pos = nombreArchivo.find(".jpg");
    // Extraer la parte del nombre sin la extensión
    std::string marcaTiempoStr = nombreArchivo.substr(0, pos);
    // Convertir el string a long long
    long long marcaTiempo = std::stoll(marcaTiempoStr);

    
		if ((!recortar_tronco(image, image)) || (buscarDistanciaCercana(marcaTiempo) > 200)) {
			total = 0;
			continue;
		}
		if (total == N_ULT_ARBOLES)
			continue;

		ultimos_arboles[total].nro_arbol = arbol;
		ultimos_arboles[total].distancia = buscarDistanciaCercana(marcaTiempo);
		// ultimos_arboles[total].image = image;
		encontrar_bordes(image, marcaTiempo, &x1, &x2);
		ultimos_arboles[total].x1 = x1;
		ultimos_arboles[total].x2 = x2;
		ultimos_arboles[total].diametro = x2-x1;
		//if (total >= 3) {
		if (total == (N_ULT_ARBOLES-1)) {
			arbol++;
			for (i=0; i<N_ULT_ARBOLES; i++) {
                		std::cout << " diametro: " << ultimos_arboles[i].diametro << "  distancia: " << ultimos_arboles[i].distancia << " relacion: " << ((double)ultimos_arboles[i].diametro / (double)ultimos_arboles[i].distancia) << std::endl;
			}
                	std::cout << " :tronco detectado. " << arbol << " " << total << " " << ss.str(); 
			encontrar_bordes(image, marcaTiempo, &x1, &x2);
		}
		total++;
		// cv::imshow("ORB Keypoints", image);
		// cv::waitKey(0);

	}
}

// ----------------------------------------------------------------------------

