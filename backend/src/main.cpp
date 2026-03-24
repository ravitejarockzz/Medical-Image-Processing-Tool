#include "crow.h"

#include "processor/CPUProcessor.h"
#include "processor/CUDAProcessor.h"
#include "factory/FilterFactory.h"

#include <opencv2/opencv.hpp>
#include "thirdparty/json.hpp"

#include <fstream>
#include <sstream>
#include <unordered_map>
#include <memory>

using json = nlohmann::json;

int main() {
    crow::SimpleApp app;

#ifdef USE_CUDA
    const bool gpuAvailable = true;
#else
    const bool gpuAvailable = false;
#endif

    // ============================================================
    // Serve Frontend Files
    // ============================================================

    CROW_ROUTE(app, "/")
    ([]() {
        std::ifstream file("../frontend/index.html");
        std::stringstream buffer;
        buffer << file.rdbuf();

        crow::response res;
        res.set_header("Content-Type", "text/html");
        res.body = buffer.str();
        return res;
    });

    CROW_ROUTE(app, "/style.css")
    ([]() {
        std::ifstream file("../frontend/style.css");
        std::stringstream buffer;
        buffer << file.rdbuf();

        crow::response res;
        res.set_header("Content-Type", "text/css");
        res.body = buffer.str();
        return res;
    });

    CROW_ROUTE(app, "/app.js")
    ([]() {
        std::ifstream file("../frontend/app.js");
        std::stringstream buffer;
        buffer << file.rdbuf();

        crow::response res;
        res.set_header("Content-Type", "application/javascript");
        res.body = buffer.str();
        return res;
    });

    // ============================================================
    // GET /operations
    // ============================================================

    CROW_ROUTE(app, "/operations")
    ([&]() {
        auto& factory = FilterFactory::instance();

        json response;
        response["cpu_operations"] = factory.getAvailableCPU();
        response["gpu_operations"] = factory.getAvailableCUDA();
        response["gpu_available"] = gpuAvailable;

        crow::response res;
        res.set_header("Content-Type", "application/json");
        res.body = response.dump();
        return res;
    });

    // ============================================================
    // POST /process
    // ============================================================

    CROW_ROUTE(app, "/process").methods("POST"_method)
    ([&](const crow::request& req) {

        try {
            crow::multipart::message multipart(req);

        // -------------------------
        // Extract Image File
        // -------------------------
        auto filePart = multipart.get_part_by_name("file");

        if (filePart.body.empty()) {
            return crow::response(400, "No file uploaded");
        }

        std::vector<uchar> buffer(
            filePart.body.begin(),
            filePart.body.end()
        );

        cv::Mat input = cv::imdecode(buffer, cv::IMREAD_COLOR);

        if (input.empty()) {
            return crow::response(400, "Invalid image");
        }

        // -------------------------
        // Extract Config JSON
        // -------------------------
        auto configPart = multipart.get_part_by_name("config");

        if (configPart.body.empty()) {
            return crow::response(400, "Missing config");
        }

        json config = json::parse(configPart.body);

            std::string operation = config["operation"];
            std::string mode = config["mode"];

            std::unordered_map<std::string, double> parameters;

            if (config.contains("parameters")) {
                for (auto& [key, value] : config["parameters"].items()) {
                    parameters[key] = value.get<double>();
                }
            }

            // -------------------------
            // Select Processor
            // -------------------------
            std::unique_ptr<ImageProcessor> processor;

            if (mode == "cuda") {
#ifdef USE_CUDA
                processor = std::make_unique<CUDAProcessor>();
#else
                return crow::response(400, "CUDA not available");
#endif
            } else {
                processor = std::make_unique<CPUProcessor>();
            }

            // -------------------------
            // Process
            // -------------------------
            cv::Mat output = processor->process(
                input,
                operation,
                parameters
            );

            // -------------------------
            // Encode PNG
            // -------------------------
            std::vector<uchar> encoded;
            cv::imencode(".png", output, encoded);

            crow::response res;
            res.code = 200;
            res.set_header("Content-Type", "image/png");
            res.body = std::string(encoded.begin(), encoded.end());

            return res;
        }
        catch (const std::exception& e) {
            return crow::response(500, e.what());
        }
    });

    // ============================================================
    // Run Server
    // ============================================================

    //app.port(18080).multithreaded().run(); // To bind to all interfaces, use app.bindaddr("0.0.0.0")
    // Binding to all interfaces

    app.bindaddr("0.0.0.0").port(18080).multithreaded().run();

    return 0;
}