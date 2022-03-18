
package com.mycompany.server.clienthandler;

/**
 *
 * @author mo
 */
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Scanner;

public class client_handler implements Runnable {
    private final Socket currentClient;
    private final List<Socket> clientsList;
    private String line = "Join to This Server";
    private volatile boolean isSent;
   int number;
    public client_handler(Socket currentClient, List<Socket> clientsList ,int client_number) {
        number=client_number;
        this.currentClient = currentClient;
        this.clientsList = clientsList;
    }

    @Override
    public void run() {
        try (Scanner in = new Scanner(currentClient.getInputStream(), StandardCharsets.UTF_8)) {
            Thread receivingThread = new Thread(() -> {
                while (true) {
                    if (in.hasNextLine()) {
                        line = in.nextLine();
                        isSent = Boolean.FALSE;
                    }
                }
            });

            receivingThread.start();
            while (true) {
                if (!isSent) {
                    clientsList.stream().filter((client) -> client != currentClient).forEach(client -> {
                        PrintWriter out = null;
                        try {
                            out = new PrintWriter(new OutputStreamWriter(client.getOutputStream(),
                                    StandardCharsets.UTF_8), true);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                        out.println("client "+number+" : " + line);
                    });
                    isSent = Boolean.TRUE;
                }
            }

        } catch (IOException exception) {
            exception.printStackTrace();
        }
    }
}