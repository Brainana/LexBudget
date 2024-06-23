# Usage:
# python scripts/record_docs_scraper_bfs.py --url https://host/path --directory /path/to/save/download/files/ --depth number --fileLinkPattern /doc/link/pattern --sameSiteFileOnly boolean
# e.g. python scripts/record_docs_scraper_bfs.py --url https://records.lexingtonma.gov/weblink --directory ./record_docs --depth 4 --fileLinkPattern /weblink/0/edoc/ --sameSiteFileOnly True
# e.g. python scripts/record_docs_scraper_bfs.py --url https://www.lexingtonma.gov/363/Annual-Budgets --directory ./budget_docs --depth 2 --fileLinkPattern /DocumentCenter/View/ --sameSiteFileOnly True
#
# This script will:
# 1. Scrape the web page for anchor tags.
# 2. Download the linked document if the anchor tag points to one.
# 3. If the anchor tag does not point to a document, recursively navigate to the linked web page and scrape documents for download up to the specified depth level.

# Import libraries
import requests  # Used to make HTTP requests
from bs4 import BeautifulSoup  # Used to scrape by HTML tag
import argparse  # Used to parse arguments
import os  # Used to create directories to save the download files
from urllib.parse import urlparse  # Used to parse url
from pathvalidate import (
    sanitize_filepath,
    sanitize_filename,
)  # Used to generate valid directory path and file name for downloading files
from queue import Queue  # Used to queue up all the links that still need to be visited
import logging  # Used to log errors

# Configure the logging
logging.basicConfig(
    filename="record_docs_scraper_errors.log",
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s]: %(message)s",
)

# Get all the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--url", type=str, required=True)
parser.add_argument("--directory", type=str, required=True)
parser.add_argument("--depth", type=int, required=True)
parser.add_argument("--fileLinkPattern", type=str, required=True)
parser.add_argument("--sameSiteFileOnly", type=bool, required=True)
args = parser.parse_args()

# Get the root URL and domain
parsed_url = urlparse(args.url)
rootUrl = "{uri.scheme}://{uri.netloc}".format(uri=parsed_url)
domain = parsed_url.netloc.replace("www.", "")
# print("Root Url: " + rootUrl + ", Domain: ", domain)

# Keep track of already visited links to avoid downloading duplicate files
visitedLinks = []

# Put links that need to be visited in queue
linksToVisit = Queue()


class LinkToVisit:
    def __init__(
        self,
        url,  # link to visit
        directory,  # directory to save the files on the link page
        depth,
    ):  # the depth relates to the original url argument for the program
        self.url = url
        self.directory = directory
        self.depth = depth


def scrape_web_page(linksToVisit, visitedLinks):
    while not linksToVisit.empty():

        # Get the first link to visit from the queue
        link = linksToVisit.get()
        url = link.url
        directory = link.directory
        depth = link.depth

        if link.depth > args.depth:
            # If the depth is already exceeded the user specified depth, just return
            return

        print("url: " + url + ", directory: " + directory + ", depth: " + str(depth))

        response = None
        try:
            # Attempt to make the HTTP request
            response = requests.get(url)

            # Check if the request was successful (status code 200)
            if response.status_code != 200:
                # Log failure
                logging.error(
                    f"Failed to request {url}: {response.status_code} - {response.reason}",
                    exc_info=True,
                )
                return
        except requests.exceptions.RequestException as e:
            # Log exception
            logging.error(f"Failed to request {url}: {e}", exc_info=True)
            return

        # Parse text
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all hyperlinks present on the webpage
        links = soup.find_all("a")

        # # For debug purposes
        # # Define the substrings to filter by
        # substrings = (
        #     "0/fol/226609/Row1.aspx",
        #     "0/fol/12488/Row1.aspx",
        #     "0/fol/511008/Row1.aspx",
        #     "0/edoc/511995/2020%20Annual%20Report.pdf",
        # )

        # # Filter the links
        # filtered_links = [
        #     link for link in links if any(sub in link.get("href", "") for sub in substrings)
        # ]

        for link in links:
            currentLink = link.get("href", "")

            if not currentLink.startswith("http"):
                # Deal with relative url
                if currentLink.startswith("#"):
                    # Skip hash tags
                    continue

                # A path that starts with a "/" is a root-relative path. It is relative to the root of the website.
                if currentLink.startswith("/"):
                    # The file link is a relative path, construct the absolute url by prefixing it with rootUrl
                    currentLink = rootUrl + currentLink
                else:
                    # A path that does not start with a "/" is a document-relative path.
                    # It is relative to the current document's location.
                    currentDocLocation = args.url
                    if not url.endswith("/"):
                        currentDocLocation += "/"
                    currentLink = currentDocLocation + currentLink

            # Avoid downloading duplicates
            if currentLink in visitedLinks:
                # print("Already visited: " + fileLink)
                continue

            # print (fileLink)

            visitedLinks.append(currentLink)

            # Only download file with a certain link pattern
            # For example: Lexington town documents are all stored in "DocumentCenter"
            # Only download Lexington town documents
            if args.fileLinkPattern in currentLink:
                print("Downloading file: ", currentLink + " : " + link.text)

                # Get response object for link
                response = requests.get(currentLink)

                # Create the directory to store files on this page if it doesn't exist yet
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # Remove all invalid charcters
                file_name = sanitize_filename(link.text)

                # Create the full path by combining the folder path and file name
                file_path = f"{directory}/{file_name}.pdf"
                print(file_path)

                # Write pdf
                pdf = open(file_path, "wb")
                pdf.write(response.content)
                pdf.close()
            elif (
                depth + 1 <= args.depth
            ):  # Only visit the link if the depth is not exceed the user specified depth
                # Skip the link for the web page that doesn't belong to the user specified web site.
                if args.sameSiteFileOnly and currentLink.find(domain) == -1:
                    continue

                print("Go into: " + currentLink)

                # Remove all invalid directory characters
                subDirName = sanitize_filepath(link.text)

                # Avoid double slashes
                subDirName = subDirName.lstrip("/")

                # Form the sub directory path to store the files on the linked page.
                subDirPath = f"{directory}/{subDirName}"

                # Add to the LinkToVisit queue
                subLink = LinkToVisit(
                    url=currentLink, directory=subDirPath, depth=depth + 1
                )
                linksToVisit.put(subLink)


rootLink = LinkToVisit(url=args.url, directory=args.directory, depth=1)
linksToVisit.put(rootLink)

scrape_web_page(linksToVisit, visitedLinks)

print("All PDF files downloaded")
