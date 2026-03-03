#include "crow.h"
#include "ImageProcessor.h"
#include "CPUProcessor.h"
#include "CUDAProcessor.h"

#include <fstream>
#include <sstream>

using namespace std;

// -----------------------------
// Utility: Read file into string
// -----------------------------
std::string readFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) return "";

    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

int main()
{
    crow::SimpleApp app;

    // -----------------------------
    // Serve static index.html
    // -----------------------------
    CROW_ROUTE(app, "/")
    ([]() {
        auto content = readFile("../static/index.html");
        crow::response res;
        res.code = 200;
        res.set_header("Content-Type", "text/html");
        res.write(content);
        return res;
    });

    // -----------------------------
    // GET /operations
    // -----------------------------
    CROW_ROUTE(app, "/operations")
    ([]() {
        crow::json::wvalue x;
        x["available_operations"][0] = "gaussian_blur";
        x["available_operations"][1] = "canny";
        x["available_operations"][2] = "grayscale";
        return x;
    });

    // -----------------------------
    // POST /process
    // -----------------------------
    CROW_ROUTE(app, "/process").methods("POST"_method)
    ([](const crow::request& req) {

        try
        {
            crow::multipart::message msg(req);

            auto filePart   = msg.get_part_by_name("file");
            auto configPart = msg.get_part_by_name("config");

            std::string imageData = filePart.body;
            std::string configStr = configPart.body;

            auto configJson = crow::json::load(configStr);
            if (!configJson)
                return crow::response(400, "Invalid JSON");

            std::string operation = configJson["operation"].s();

            std::string mode = configJson["mode"].s();

            std::unique_ptr<ImageProcessor> processor;

            if (mode == "cuda")
                processor = std::make_unique<CUDAProcessor>();
            else
                processor = std::make_unique<CPUProcessor>();

            std::vector<unsigned char> result;

            if (operation == "gaussian_blur")
            {
                int kernel = configJson["parameters"]["kernel"].i();
                result = processor->applyGaussianBlur(imageData, kernel);
            }
            else if (operation == "canny")
            {
                int low  = configJson["parameters"]["low"].i();
                int high = configJson["parameters"]["high"].i();
                result = processor->applyCanny(imageData, low, high);
            }
            else if (operation == "grayscale")
            {
                result = processor->applyGrayscale(imageData);
            }
            else
            {
                return crow::response(400, "Unknown operation");
            }

            crow::response res;
            res.code = 200;
            res.set_header("Content-Type", "image/png");
            res.body = std::string(result.begin(), result.end());
            return res;
        }
        catch (const std::exception& e)
        {
            return crow::response(400, "Multipart parsing failed");
        }
    });

    // -----------------------------
    // Serve static CSS & JS
    // -----------------------------
    CROW_ROUTE(app, "/<string>")
    ([](const std::string& filename) {
        std::string path = "../static/" + filename;
        auto content = readFile(path);

        crow::response res;

        if (filename.find(".css") != std::string::npos)
            res.set_header("Content-Type", "text/css");
        else if (filename.find(".js") != std::string::npos)
            res.set_header("Content-Type", "application/javascript");

        res.write(content);
        return res;
    });

    // -----------------------------
    // Start server
    // -----------------------------
    app.port(18080)
       .multithreaded()
       .run();
}