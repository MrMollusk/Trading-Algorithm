#include <iostream>
#include <vector>
#include <functional>
using namespace std;

void func1() {
	cout << "Function1\n";
}

void func2() {
	cout << "Function2\n";
}

void func3() {
	cout << "Function3\n";
}

int main() {
	vector<function<void()>> funcs = {func1, func2, func3};
}

for (auto& func : funcs) {
	func();
}