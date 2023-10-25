# GraphicalEditorForNN

## Links

[Figma](https://www.figma.com/file/VlSKVSf3cpgZ1pa75CTaMb/Untitled?type=design&node-id=0-1&mode=design&t=kecMaQTEdpRHFw8j-0)

[Ugile project](https://ru.yougile.com/team/b400e1850fe9/GraphicalEditorForNN) (authorization required)

[Schedule table](https://docs.google.com/spreadsheets/d/1BtKyKgk-_1t9loRz4vYTFROSOF-8Fd3Q9gN2qE21gpA/edit?usp=sharing)

[Retrospectives](https://docs.google.com/spreadsheets/d/1N3NUDa-gbqLRaJE3SnSPSLX4hekyoULw-SIBjqRFItg/edit?usp=sharing)

## Build options

There is one Makefile at the project root and a Makefile for each folder (`client`, `py_server`, `server`)

For each Makefile you can run 
```
make help
```
It will give a list of all available **targets** and **options** for the current Makefile

Common Options (`make <something> option1=value1 option2=value2 ...`):
- V: verbose level, if you want `make` to print what it's doing use `V=2`
- CONFIG_PATH: path to the `config.json`, default is the right one

Also, if you need some other options, remember that you can override any variable used in the Makefile using an option 
with that name.

### Run Client
Client has only one target. You can also provide options listed in the `make help` for example:
```
cd client
make port=2002 V=2
```
Runs server on the port 2002, or just use `make` to run it on the default port

> **Important:** `npx serve` is used to host the client code (`index.html`) on the *port* specified in `config.json`  
> Python server is configured to only accept requests from this *port*  
> If you run your `index.html` as a file and will send requests to python server, `CORS` won't allow you to do it  
> If you want to make it simple and allow Python server to accept anything, uncomment line at
> [`__init__.py:41`](py_server/mlcraft/__init__.py?plain=1#L41)

### Run Python Server
Make sure you have `python` version at least **3.10**  
Then a simple `make` in the `py_server` directory should work  
It will build you a virtual environment in the folder `py_server/.venv`. Then it will install all dependencies in it
via `pip`, including our own project `mlcraft`.  
After that, it launches `mlcraft` using `flask` at specified port (see `make help` for all available options here).
```
cd py_server
make 
```
Other targets available:  
- `make format`: formats your code using `black` formatter, use before commit, **CI won't let you pass with unformatted code**
- `make test`: launch `mypy` typechecking and all tests from `tests` directory using `pytest`  
- `make clean`: deletes current instance folder of the app (I don't think you'll need to use it ever)  

### Run C++ Server

> **Important:** If you want to build a c++ server, you need to install [Boost](https://www.boost.org/users/download/).  
> Then, add the path to the boost root (folder with `include` and `lib` inside) in the `config.json` file.  
> For example: `"BOOST_ROOT": "/usr/local/Cellar/boost/1.81.0_1"`  
> After that you should be able to build everything just fine...

There are 3(4) main targets available to build:
- `make`: runs the `core/main.cpp` file (use it to test you Tensor and so on...)
- `make serve`: runs the server (`api/server.cpp`)
- `make test`: runs all the tests in the `tests` directory (test file for X.cpp **must** be named `XTests.cpp`)
- `make test.X`: runs a single test for `X`

Examples:
```
cd server
make serve port=4000
make test.Blob V=2 O=2
```

To change the compiler flags you can edit some Makefile lines about the `CXXFLAGS` variable

### Run from root folder

You can build and test things from the root folder as well, but I don't think it is useful. It's done only to 
conveniently run the [CI](.github/workflows/CI.yml)   
Use `make help` to see available targets there

## Formats

Some json formats to use for requests

[`Server++/train`](documentation/api-examples/train.json)  

[`Server++/predict`](documentation/api-examples/predict.json)  

[`Server/predict`](documentation/api-examples/userPredict.json)  

## Technology stack

### Languages:
 - C++
 - Python
 - Javascript 

### Server++:
 - Network: [Crow](https://github.com/CrowCpp/Crow)
 - Build: [Makefile](https://www.gnu.org/software/make/manual/make.html)
  
### Server:
 - [Flask](https://flask.palletsprojects.com/en/3.0.x/)
 - [SQLAlchemy](https://flask-sqlalchemy.palletsprojects.com/en/3.1.x/)
  
![](documentation/interaction.png)

------------------

## Architecture

### Overview
![](documentation/ComponentsArchitecture.jpg)

### Server++
![](documentation/ServerArchitecture.svg)

### Data
![](documentation/DatabaseArchitecture.jpg)
