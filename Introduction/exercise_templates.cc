#include <iostream>


template<typename T> struct TestClass{
  T x;

  TestClass(T const& x) : x(x){}
  operator T() const{ return x; }
};
template<> struct TestClass<float>{
  float x;

  TestClass(float const& x) : x(x+1e-3f){}
  operator float() const{ return x; }
};
template<typename T> struct TestClass<TestClass<T>>{
  T x;

  TestClass(TestClass<T> const& x) : x(-x){}
  operator T() const{ return x; }
};

template<typename T> void print_value(T const& x){
  std::cout << "Value is " << x << "." << std::endl;
}
template<> void print_value(float const& x){
  std::cout << "Twice the value of the float " << x << " is " << 2*x << "." << std::endl;
}


int main(){
  print_value(5); // int
  print_value(2.3); // double
  print_value(5.1f); // float

  // The following lines will call print_value(TestClass<T> const& x) even if operator T() is defined!
  print_value(TestClass<double>(7.25)); // struct initialization with a double type argument
  print_value(TestClass<float>(1.22f)); // struct initialization with a float type argument

  // This is the correct way to force the call print_value(float const& x).
  print_value((float) TestClass<float>(1.22f));

  TestClass<double> cx(87);
  print_value(TestClass<TestClass<double>>(cx));
}
