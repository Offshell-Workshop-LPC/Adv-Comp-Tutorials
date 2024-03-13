#include <iostream>
#include <utility>


void print_test(int n){
  using namespace std;
  cout << "Pass by value: " << n << ", address: " << &n << endl;
}
void print_test(double& d){
  using namespace std;
  cout << "Pass by non-const lvalue reference: " << d << ", address: " << &d << endl;
  d = 7.2;
}
void print_test(double const& d){
  using namespace std;
  cout << "Pass by const lvalue reference: " << d << ", address: " << &d << endl;
}
void print_test(double&& d){
  using namespace std;
  cout << "Pass by non-const rvalue reference: " << d << ", address: " << &d << endl;
  d = 9.3;
}
void print_test(double const&& d){
  using namespace std;
  cout << "Pass by const rvalue reference: " << d << ", address: " << &d << endl;
  //d = 11.2; // Error: d is const
}

void print_test(int const* n){
  if (n == nullptr) std::cout << "nullptr (pointer to constant)" << std::endl;
  else std::cout << "Passed pointer " << n << " to constant value *n = " << *n << std::endl;
  //*n = 9; // Error: n is const
}
void print_test(int* const& n){
  if (n == nullptr) std::cout << "nullptr (constant reference to pointer)" << std::endl;
  else std::cout << "Passed a constant reference to a pointer " << n << " with value *n = " << *n << std::endl;
  *n = 9; // This WILL work! Why?
}

int main(){
  using namespace std;

  int n = 3;
  cout << "Value of n: " << n << ", address of n : " << &n << endl;
  print_test(n);
  print_test(5);
  int* ptr_n = &n;
  cout << "Value of ptr_n: " << ptr_n << ", value of its object: " << *ptr_n << endl;
  print_test(ptr_n);
  int const& ref_n = n;
  cout << "Value of ref_n: " << ref_n << ", address of ref_n: " << &ref_n << endl;
  print_test(&ref_n);

  double d = 3.14;
  cout << "Value of d: " << d << ", address of d: " << &d << endl;
  print_test(d);

  double const dc = 2.3;
  cout << "Value of dc: " << dc << ", address of dc: " << &dc << endl;
  print_test(dc);

  print_test(4.5);

  print_test(std::move(d));
  cout << "Value of d (= 3.14) after call with std::move(d): " << d << ", address: " << &d << endl;

  print_test(std::move(dc));
  cout << "Value of dc (= 2.3) after call with std::move(dc): " << dc << ", address: " << &dc << endl;
}
