import java.util.Scanner;

public class Snippet {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        if (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            System.out.print(line); // write the exact line back
        }
        scanner.close();
    }
}
