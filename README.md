# GraphicalEditorForNN

## Build options
> **Note**: If you want to build a c++ server, you need to install [Boost](https://www.boost.org/users/download/)
> Then, add an environment variable pointing to the boost root folder (folder with `include` and `lib` inside).
> For example:
> ```bash
> export BOOST_ROOT="/usr/local/Cellar/boost/1.81.0_1/"
> ```
> After that you should be able to build everything just fine...


1. **Quick and dirty:** Build everything and launch in debug mode
```
make
```
2. **Official**: Build everything and launch in release mode *(the only significant difference for now is that c++ code will run faster)*
```
make release
```
This way, it will be a total mess in the terminal. Need the logging files

3. **Selective:** Run separately each component, for example for `py_server`
```
cd py_server
make
```
This way, only one component will be launched in the debug mode

4. **Run tests:** The same thing can be done with tests, you can either run
```
make test
```
In the root, or separately in each directory *(currently frontend doesn't have any tests)*

## Formats

Some json formats to use for requests

[`Server++/train`](documentation/api-examples/train.json)  

[`Server++/predict`](documentation/api-examples/predict.json)  

[`Server/predict`](documentation/api-examples/userPredict.json)  

## Links

[Figma](https://www.figma.com/file/VlSKVSf3cpgZ1pa75CTaMb/Untitled?type=design&node-id=0-1&mode=design&t=kecMaQTEdpRHFw8j-0)

[Ugile project](https://ru.yougile.com/team/b400e1850fe9/GraphicalEditorForNN) (authorization required)

[Schedule table](https://docs.google.com/spreadsheets/d/1BtKyKgk-_1t9loRz4vYTFROSOF-8Fd3Q9gN2qE21gpA/edit?usp=sharing)

[Sprint 1 Retrospective](https://docs.google.com/spreadsheets/d/1N3NUDa-gbqLRaJE3SnSPSLX4hekyoULw-SIBjqRFItg/edit?usp=sharing)

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
